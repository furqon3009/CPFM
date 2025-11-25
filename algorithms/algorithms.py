# import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.models import Temporal_Imputer, masking, AutoEncoder, feat_bootleneck, Classifier, TimeCLREncoder, Transformer, DomainClassifier
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from utils import RMSELoss, get_distances, soft_k_nearest_neighbors, refine_predictions, eval_and_label_dataset, update_labels
import random
from itertools import compress
from sklearn.metrics import accuracy_score
from torchmetrics import Accuracy, AUROC, F1Score
# from trainers.abstract_trainer import calculate_metrics


from moment.momentfm.models.moment import MOMENTPipeline


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class SHOT(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(SHOT, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # Freeze the classifier
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # obtain pseudo labels for each epoch
            pseudo_labels = self.obtain_pseudo_labels(trg_dataloader)

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                # prevent gradient accumulation
                self.optimizer.zero_grad()

                # Extract features
                trg_feat, _ = self.feature_extractor(trg_x)
                trg_pred = self.classifier(trg_feat)

                # pseudo labeling loss
                pseudo_label = pseudo_labels[trg_idx.long()].to(self.device)
                target_loss = F.cross_entropy(trg_pred.squeeze(), pseudo_label.long())

                # Entropy loss
                softmax_out = nn.Softmax(dim=1)(trg_pred)
                entropy_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(softmax_out))

                #  Information maximization loss
                entropy_loss -= self.hparams['im'] * torch.sum(
                    -softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))

                # Total loss
                loss = entropy_loss + self.hparams['target_cls_wt'] * target_loss

                # self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses = {'Total_loss': loss.item(), 'Target_loss': target_loss.item(),
                          'Ent_loss': entropy_loss.detach().item()}

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

    def obtain_pseudo_labels(self, trg_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        preds, feas = [], []
        with torch.no_grad():
            for inputs, labels, _ in trg_loader:
                inputs = inputs.float().to(self.device)

                features, _ = self.feature_extractor(inputs)
                predictions = self.classifier(features)
                preds.append(predictions)
                feas.append(features)

        preds = torch.cat((preds))
        feas = torch.cat((feas))

        preds = nn.Softmax(dim=1)(preds)
        _, predict = torch.max(preds, 1)

        all_features = torch.cat((feas, torch.ones(feas.size(0), 1).to(self.device)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        K = preds.size(1)
        aff = preds.float().cpu().numpy()
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = torch.from_numpy(pred_label)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = torch.from_numpy(pred_label)

        self.feature_extractor.train()
        self.classifier.train()
        return pred_label

class AaD(Algorithm):
    """
    (NeurIPS 2022 Spotlight) Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation
    https://github.com/Albert0147/AaD_SFDA
    """

    def __init__(self, backbone, configs, hparams, device):
        super(AaD, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # inilize alpha value

            # defining best and last model
            best_src_risk = float('inf')
            best_model = self.network.state_dict()
            last_model = self.network.state_dict()

            fea_bank, score_bank = self.build_feat_score_bank(trg_dataloader)

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                num_samples = len(trg_dataloader.dataset)

                # Extract features
                features, _ = self.feature_extractor(trg_x)
                predictions = self.classifier(features)

                # output softmax probs
                softmax_out = nn.Softmax(dim=1)(predictions)

                alpha = (1 + 10 * step / self.hparams["num_epochs"] * len(trg_dataloader)) ** (-self.hparams['beta']) * \
                        self.hparams['alpha']
                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                # start gradients
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))

                mask = torch.ones((trg_x.shape[0], trg_x.shape[0]))
                diag_num = torch.diag(mask)
                mask_diag = torch.diag_embed(diag_num)
                mask = mask - mask_diag
                copy = softmax_out.T  # .detach().clone()#

                dot_neg = softmax_out @ copy  # batch x batch

                dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
                neg_pred = torch.mean(dot_neg)
                loss += neg_pred * alpha

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(loss.item(), 32)

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        return last_model, best_model

    def build_feat_score_bank(self, data_loader):
        fea_bank = torch.empty(0).cuda()
        score_bank = torch.empty(0).cuda()

        self.feature_extractor.eval()
        self.classifier.eval()

        # Process data batch by batch
        for data in data_loader:
            batch_data = data[0].cuda()  # Assuming the first element in your batch is the data. Adjust as needed.

            batch_feat,_ = self.feature_extractor(batch_data)
            norm_feats = F.normalize(batch_feat)
            batch_pred = self.classifier(batch_feat)
            batch_probs = nn.Softmax(dim=-1)(batch_pred)

            # Update the banks
            fea_bank = torch.cat((fea_bank, norm_feats.detach()), 0)
            score_bank = torch.cat((score_bank, batch_probs.detach()), 0)

        return fea_bank, score_bank

class NRC(Algorithm):
    """
    Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation (NIPS 2021)
    https://github.com/Albert0147/NRC_SFDA
    """

    def __init__(self, backbone, configs, hparams, device):
        super(NRC, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                # Extract features
                features, _ = self.feature_extractor(trg_x)
                predictions = self.classifier(features)
                num_samples = len(trg_dataloader.dataset)
                fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                score_bank = torch.randn(num_samples, self.configs.num_classes).cuda()
                softmax_out = nn.Softmax(dim=1)(predictions)

                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.cpu().detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                    fea_near = fea_bank[idx_near]  # batch x K x num_dim
                    fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
                    distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                    _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                                  k=5 + 1)  # M near neighbors for each of above K ones
                    idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                    trg_idx_ = trg_idx.unsqueeze(-1).unsqueeze(-1)
                    match = (
                            idx_near_near == trg_idx_).sum(-1).float()  # batch x K
                    weight = torch.where(
                        match > 0., match,
                        torch.ones_like(match).fill_(0.1))  # batch x K

                    weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                            5)  # batch x K x M
                    weight_kk = weight_kk.fill_(0.1)

                    # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                    # weight_kk[idx_near_near == trg_idx_]=0

                    score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                    # print(weight_kk.shape)
                    weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                            -1)  # batch x KM

                    score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                                    self.configs.num_classes)  # batch x KM x C

                    score_self = score_bank[trg_idx]

                # start gradients
                output_re = softmax_out.unsqueeze(1).expand(-1, 5 * 5,
                                                            -1)  # batch x C x 1
                const = torch.mean(
                    (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                     weight_kk.cuda()).sum(
                        1))  # kl_div here equals to dot product since we do not use log for score_near_kk
                loss = torch.mean(const)

                # nn
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss += torch.mean(
                    (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

                # self, if not explicitly removing the self feature in expanded neighbor then no need for this
                # loss += -torch.mean((softmax_out * score_self).sum(-1))

                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax *
                                          torch.log(msoftmax + self.hparams['epsilon']))
                loss += gentropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(loss.item(), 32)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class MAPU(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class DINE(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(DINE, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.feat_bootleneck = feat_bootleneck(configs, bottleneck_dim=256, type="bn")

        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.network2 = nn.Sequential(self.feature_extractor, self.feat_bootleneck, self.classifier)


        self.optimizer = torch.optim.SGD(
            self.network2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network2.state_dict()
        last_model = self.network2.state_dict()

        
        param_group = []
        learning_rate = self.hparams["learning_rate"]

        for k, v in self.feature_extractor.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate*0.1}]
        for k, v in self.classifier.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}] 

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                

                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network2.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class DINE2_source(Algorithm):

    def __init__(self, configs, hparams, device):
        super(DINE2_source, self).__init__(configs)

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        # print(f'self.sourceModel:{self.sourceModel}')

        # self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module

        self.pre_optimizer = torch.optim.Adam(
            self.sourceModel.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # device
        self.device = device
        self.hparams = hparams

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _, _, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                # print(f'src_x:{src_x}')

                # forward pass correct sequences
                outputs = self.sourceModel(src_x)

                # classifier predictions
                src_pred = outputs.logits

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.sourceModel.state_dict())
        return src_only_model

class DINE2_target(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(DINE2_target, self).__init__(configs)

        self.best_model_net1 = True
        self.pretrained_source = False

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module


        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )


        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def update(self, trg_dataloader, avg_meter, logger, source_model_dir):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = self.modelmoment.state_dict()
        self.last_model = self.modelmoment.state_dict()
        self.source_model_dir = source_model_dir
        # print(f'source_model_dir:{source_model_dir}')

        # print(f'self.pretrained_source:{self.pretrained_source}')

        if not self.pretrained_source:
            print(f'Load pretrained source model..')
            load_source_model_path = source_model_dir + "/checkpoint.pt"
            print(f'source model path:{load_source_model_path}')
            self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
            for param in self.sourceModel.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')

        total_epochs = self.hparams["num_epochs"] 
        
        self.sourceModel.eval()

        # train
        for epoch in range(0, total_epochs):
        
            self.modelmoment.eval()

            for step, (trg_x, _, trg_idx, _, _) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)

                start_test = True
                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    _, src_idx = torch.sort(src_prob, 1, descending=True)
                    # print(f'src_idx:{src_idx}')
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)

                    if self.hparams["topk"] > 0:
                        topk = np.min([self.hparams["topk"], self.configs.num_classes])
                        for i in range(src_prob.size()[0]):
                            src_prob[i, src_idx[i, topk:]] = (1.0 - src_prob[i, src_idx[i, :topk]].sum())/ (src_prob.size()[1] - topk)

                    if start_test:
                        all_src_prob = src_prob.float()
                        start_test = False
                    else:
                        all_src_prob = torch.cat((all_src_prob, src_prob.float()), 0)
                    
                    mem_P = all_src_prob.detach()


                if args.ema < 1.0 and iter_num > 0 and iter_num % interval_iter == 0:
                    start_test = True
                    with torch.no_grad():
                        # iter_test = iter(dset_loaders["target_te"])
                        # for i in range(len(dset_loaders["target_te"])):
                        for i in range(len(trg_x)):
                            data = iter_test.next()
                            inputs = data[0]
                            inputs = inputs.cuda()
                            outputs = model(inputs)
                            outputs = nn.Softmax(dim=1)(outputs)
                            if start_test:
                                all_output = outputs.float()
                                start_test = False
                            else:
                                all_output = torch.cat((all_output, outputs.float()), 0)
                        mem_P = mem_P * args.ema + all_output.detach() * (1 - args.ema)
                    model.train()



                # Target Model1 output
                outputs = self.modelmoment(trg_x)
                trg_pred = outputs.logits

                CE_loss = self.cross_entropy(trg_pred, pseudo_labels)

                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                # loss = loss_1 + AE_loss + loss_reconstruct
                loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses = {'entropy_loss': CE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                #  'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()


            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

class B2TSDA(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA, self).__init__(configs)

        # self.feature_extractor = backbone(configs)
        # self.classifier = classifier(configs)
        # self.temporal_verifier = Temporal_Imputer(configs)

        self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls = nn.DataParallel(self.AE_cls)
        self.AE_cls = self.AE_cls.module

        self.AE_cls2 = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        self.AE_cls2 = self.AE_cls2.module

        self.best_model_net1 = True
        self.pretrained_source = False

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        self.modelmoment2 = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment2.init()

        # print(f'self.sourceModel:{self.sourceModel}')

        # self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module
        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module
        self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        self.modelmoment2 = self.modelmoment2.module
        self.AE_cls = nn.DataParallel(self.AE_cls)
        self.AE_cls = self.AE_cls.module
        self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        self.AE_cls2 = self.AE_cls2.module


        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.modelmoment2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE = torch.optim.Adam(
            self.AE_cls.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE2 = torch.optim.Adam(
            self.AE_cls2.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.sourceModel.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()

                # forward pass correct sequences
                outputs = self.sourceModel(src_x)

                # classifier predictions
                src_pred = outputs.logits

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.sourceModel.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger, source_model_dir):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = self.modelmoment.state_dict()
        self.last_model = self.modelmoment.state_dict()
        self.source_model_dir = source_model_dir

        # Models moment trainable parameters
        for name, param in self.modelmoment.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        # for k, v in self.classifier.named_parameters():
        # freeze both classifier and ood detector
        # for k, v in self.classifier.named_parameters():
        #     v.requires_grad = False
        # for k, v in self.temporal_verifier.named_parameters():
        #     v.requires_grad = False

        # print(f'self.pretrained_source:{self.pretrained_source}')

        if not self.pretrained_source:
            print(f'Load pretrained source model..')
            load_source_model_path = source_model_dir + "/checkpoint.pt"
            print(f'source model path:{load_source_model_path}')
            self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
            for param in self.sourceModel.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')

        total_epochs = self.hparams["num_epochs"] 
        forget_rate = np.ones(total_epochs) * self.hparams["forget_rate"]
        forget_rate[:(self.hparams["warm_target"]+self.hparams["num_gradual"])] = np.linspace(0, self.hparams["forget_rate"], (self.hparams["warm_target"]+self.hparams["num_gradual"]))
        # print(f'forget_rate:{forget_rate}')

        # train
        for epoch in range(1, total_epochs+1):
        # for epoch in range(1, self.hparams["num_epochs"] + 1):
            if epoch <= round(total_epochs/2):#total_epochs
                alpha = 0.0
            else:
                alpha = (epoch*2-total_epochs)/total_epochs

            
            self.modelmoment.train()
            self.modelmoment2.train()
            # self.network.train()
            # self.network2.train()
            self.AE_cls.train()
            self.AE_cls2.train()

            Total_loss_network = self.target_train(trg_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, self.AE_cls, self.optimizerAE, epoch, logger)

            Total_loss_network2 = self.target_train(trg_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, self.AE_cls2, self.optimizerAE2, epoch, logger)

            

            self.lr_scheduler.step()
            self.lr_scheduler2.step()
            self.lr_schedulerAE.step()
            self.lr_schedulerAE2.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and (Total_loss_network.avg < best_src_risk or Total_loss_network2.avg < best_src_risk):
                # best_src_risk = avg_meter['Total_loss'].avg

                if Total_loss_network.avg < Total_loss_network2.avg:
                    best_src_risk = Total_loss_network.avg
                    self.best_model = deepcopy(self.modelmoment.state_dict())
                    self.best_model_net1 = True
                else:
                    best_src_risk = Total_loss_network2.avg
                    self.best_model = deepcopy(self.modelmoment2.state_dict())
                    self.best_model_net1 = False
            # if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
            #     best_src_risk = avg_meter['Src_cls_loss'].avg
            #     self.best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def target_train(self, trg_dataloader, avg_meter, modelmoment, optimizer, modelmoment_ind, sourceModel, forget_rate, alpha, AE_cls, optimizer_AE, epoch, logging):

        for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

            # Target data
            trg_x = trg_x.float().to(self.device)
            # trg_y = trg_y.long().to(self.device)
            # print(f'trg_x:{trg_x}')

            # val_x, val_y, _ = iter(val_dataloader).next()

            # Validation data
            # val_x, val_y, _ = next(iter(val_dataloader))
            # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)
            # print('sebelum print trg_x.shape')
            # print(f'trg_x.shape:{trg_x.shape}')
            # print(f'val_x.shape:{val_x.shape}')

            # Teacher's soft-labels and pseudo-labels
            # soft_labels_list, _, pseudo_labels_list = self.ensembl_inference(model_t_all, trg_x)
            # feat_t_list, soft_labels_list = self.ensembl_inference2(self.model_t_all, trg_x)
            
            # print(f'soft_labels_list[0].shape:{soft_labels_list[0].shape}')

            # output = sourceModel(trg_x)
            # self.model_t_all.append(sourceModel)
            # print('sourceModel ok')
            # print(f'self.model_t_all[0]:{self.model_t_all[0]}')
            # print(f'sourceModel:{sourceModel}')
            # output = self.model_t_all[5](trg_x)
            # print(f'output.logits:{output.logits}')
            # output2 = self.model_t_all[4](trg_x)
            # print(f'output2.logits:{output2.logits}')

            with torch.no_grad():
                outputs_src = self.sourceModel(trg_x)
                trg_pred_src = outputs_src.logits

                src_prob = nn.Softmax(dim=1)(trg_pred_src)
                src_conf, pseudo_labels = torch.max(src_prob, 1)
                # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)

            # print(f'src_prob:{src_prob}')
            # print(f'src_conf:{src_conf}')
            # print(f'pseudo_labels:{pseudo_labels}')
            # print(f'pseudo_labels2:{pseudo_labels2}')

            # # select pseudo-labels based on threshold -----------------------------------------------
            # conf_sel = src_conf > 0.7
            # ind_keep = list(compress(range(len(conf_sel)), conf_sel))
            # # print(f'len(trg_x):{len(trg_x)}')
            # trg_x = trg_x[ind_keep]
            # pseudo_labels = pseudo_labels[ind_keep]
            # # print(f'after ind_keep len(trg_x):{len(trg_x)}')
            # # ind_keep = torch.squeeze(conf_sel.nonzero(), dim=-1)
            # # print(f'conf_sel:{conf_sel}')
            # # print(f'ind_keep:{ind_keep}')
            # # end select pseudo-labels based on threshold ------------------------------------------

            # teacher_mask_off=[0]*self.num_of_teachers    # 1-seleced, 0-not selected

            optimizer.zero_grad()
            optimizer_AE.zero_grad()

            # print(f'self.network.parameters:{self.network.parameters}')

            # Co-training model output
            with torch.no_grad():
                outputs_ind = modelmoment_ind(trg_x)
                trg_pred_ind = outputs_ind.logits
                trg_ind_prob = nn.Softmax(dim=1)(trg_pred_ind)

            # Student's model output
            outputs = modelmoment(trg_x)
            trg_pred = outputs.logits

            # max_output, _ = torch.max(trg_pred, 1)
            # max_output_ind, _ = torch.max(trg_pred_ind, 1)

            # a = max_output/(max_output+max_output_ind)
            # b = max_output_ind/(max_output+max_output_ind)

            # # Aggregate output 
            # output_target = torch.unsqueeze(a,1)*trg_pred + torch.unsqueeze(b,1)*trg_pred_ind

            # print(f'output_target.shape:{output_target.shape}')

            # print(f'trg_pred:{trg_pred}')
            # print(f'trg_pred.shape:{trg_pred.shape}')
            # print(f'a:{a}')
            # print(f'a.shape:{a.shape}')
            # print(f'torch.unsqueeze(a,1)*trg_pred:{torch.unsqueeze(a,1)*trg_pred}')

            # with torch.no_grad():
            #     outputs_src = sourceModel(trg_x)
            #     trg_pred_src = outputs_src.logits

            #     src_prob = nn.Softmax(dim=1)(trg_pred_src)
            #     _, pseudo_labels = torch.max(src_prob, 1)

            # print(f'pseudo_labels:{pseudo_labels}')
            # print(f'trg_y:{trg_y}')

            # pseudo-label co-guessing
            # outputs_allsoft = (1 - alpha) * trg_pred_src + alpha * output_target
            # _, pseudo_labels = torch.max(outputs_allsoft, 1)

            # reconstruct -------------------------------------------------------------
            mt = random.uniform(0,1) #mask
            s0,s1,s2 = trg_x.shape
            randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
            mt = torch.bernoulli(randuniform).to(self.device)
            m_ones = torch.ones(s0,s1,s2).to(self.device)

            sum_mt = mt.flatten().sum()
            # c_mask = 1/((s0*s1*s2) * sum_mt)
            # c_unmask = 1/(((s0*s1*s2) - sum_mt)*(s0*s1*s2))
            c_mask = sum_mt/(s0*s1*s2)
            c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)
            # print(f'c_mask:{c_mask}')
            # print(f'c_unmask:{c_unmask}')
            # print(f'mt:{mt}')
            # print(f'trg_x.shape:{trg_x.shape}')

            #src if mt=1 -> x=0
            src2 = torch.clone(trg_x)
            src2 = src2 * (m_ones-mt)
            gamma = 0.5
            criterion = RMSELoss()
            # print(f'src2.shape:{src2.shape}')

            # pred reconstruct
            out = self.modelmoment.forward_reconstruct(src2)
            pred_reconstruct = out.reconstruction
            # print(f'pred_reconstruct.shape:{pred_reconstruct.shape}')
            # print(f'src2.shape:{src2.shape}')
            # print(f'criterion(pred_reconstruct, src2):{criterion(pred_reconstruct, src2)}')
            #loss reconstruct
            loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)
            # print(f'loss_reconstruct:{loss_reconstruct}')
            # --------------------------------------------------------------------------

            # co-training loss
            loss_cot = F.cross_entropy(trg_pred_ind, pseudo_labels, reduction='none') # reduction='none' to get loss per batch instead of avg loss
            # loss_cot = F.cross_entropy(output_target, pseudo_labels, reduction='none')
            # print(f'loss_cot:{loss_cot}')
            
            # get (1-R) percent low loss samples
            ind_loss_sorted = np.argsort(loss_cot.cpu().data).to(self.device)
            # ind_loss_sorted_thres = np.argsort(trg_ind_prob.cpu().data).to(self.device)
            # print(f'ind_loss_sorted:{ind_loss_sorted}')
            # print(f'ind_loss_sorted_thres:{ind_loss_sorted_thres}')
            # print(f'trg_ind_prob:{trg_ind_prob}')
            # print(f'len(ind_loss_sorted):{len(ind_loss_sorted)}')
            # print(f'len(ind_loss_sorted_thres):{len(ind_loss_sorted_thres)}')

            # Models head trainable parameters
            # for name, param in self.modelmoment.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)

            # for name, param in self.AE_cls.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
            #     else:
            #         print('no param AE_cls')


            
            # print(f'loss_cot:{loss_cot}')
            # print(f'ind_loss_sorted:{ind_loss_sorted}')
            # print(f'forget_rate:{forget_rate}')
            # print(f'alpha:{alpha}')
            remember_rate = 1 - (1-alpha) * forget_rate
            num_remember = math.ceil(remember_rate * len(ind_loss_sorted))
            ind_loss_update = ind_loss_sorted[:num_remember]
            # ind_loss_neg_update = ind_loss_sorted[(num_remember):]#/*2+num_neg

            # print(f'remember_rate:{remember_rate}')
            # print(f'len(ind_loss_update):{len(ind_loss_update)}')
            # print(f'num data:{num_remember}')
            # print(f'trg_x[ind_loss_update]:{trg_x[ind_loss_update]}')
            # outputs_sorted = modelmoment(trg_x[ind_loss_update])
            # trg_pred2 = outputs_sorted.logits

            # outputs_ind_sorted = modelmoment_ind(trg_x[ind_loss_update])
            # trg_pred_ind2 = outputs_ind_sorted.logits

            # max_output2, _ = torch.max(trg_pred2, 1)
            # max_output_ind2, _ = torch.max(trg_pred_ind2, 1)

            # a2 = max_output2/(max_output2+max_output_ind2)
            # b2 = max_output_ind2/(max_output2+max_output_ind2)
            # output_target2 = torch.unsqueeze(a2,1)*trg_pred2 + torch.unsqueeze(b2,1)*trg_pred_ind2

            
            # trg_prob_sorted = torch.log_softmax(outputs_sorted.logits, dim=1)

            # with torch.no_grad():
            #     outputs_src_sorted = sourceModel(trg_x[ind_loss_update])
            #     trg_pred_src_sorted = outputs_src_sorted.logits

            # trg_prob_src_sorted = nn.Softmax(dim=1)(trg_pred_src_sorted)
            # _, pseudo_labels_sorted = torch.max(trg_prob_src_sorted, 1)

            pseudo_labels_sorted = pseudo_labels[ind_loss_update].to(self.device)

            # with torch.no_grad():
            #     outputs_src2 = sourceModel(trg_x[ind_loss_update])
            #     trg_pred_src2 = outputs_src2.logits

            # src_prob2 = nn.Softmax(dim=1)(trg_pred_src2)
            # _, pseudo_labels2 = torch.max(src_prob2, 1)
            # print(f'pseudo_labels_sorted:{pseudo_labels_sorted}')
            # print(f'pseudo_labels2:{pseudo_labels2}')
            # print(f'trg_prob_sorted:{trg_prob_sorted}')
            # print(f'prompt.shape:{prompt.shape}')

            # Entropy loss
            # trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
            # print(f'torch.mean(EntropyLoss(trg_prob_sorted):{torch.mean(EntropyLoss(trg_prob_sorted))}')
            # trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob_sorted))

            # trg_prob_sorted = torch.log_softmax(trg_pred[ind_loss_update], dim=1).cuda()
            # print(f'trg_prob_sorted:{trg_prob_sorted}')
            # print(f'trg_prob_sorted.shape:{trg_prob_sorted.shape}')
            

            # CE_loss = self.cross_entropy(outputs_sorted.logits, pseudo_labels_sorted)
            # CE_loss = self.cross_entropy(output_target[ind_loss_update], pseudo_labels_sorted)
            CE_loss = self.cross_entropy(trg_pred[ind_loss_update], pseudo_labels_sorted)
            # print(f'CE_loss:{CE_loss}')

            # CE_loss = self.cross_entropy(trg_pred[ind_loss_update], pseudo_labels_sorted)
            # CE_loss = torch.mean(torch.sum(-1 * torch.unsqueeze(pseudo_labels_sorted,1)* trg_prob_sorted, dim=1))

            # CE_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob_sorted))
            # ind_2_sorted = np.argsort(trg_ent.cpu().data).cuda()
            # print(f'ind_2_sorted:{ind_2_sorted}')
            # print(f'trg_ent:{trg_ent}')

            # use prompt
            prompt = modelmoment.prompt.prompt
            # print(f'prompt.data:{prompt}')
            # print(f'prompt.shape:{prompt.shape}')
            # print(f'trg_pred[ind_loss_update].shape:{trg_pred[ind_loss_update].shape}')

            # prompt reconstruction -------------------------------------------------------------------------
            AE_loss = 0
            prompt_new = AE_cls(prompt)
            # print(f'prompt:{prompt}')
            # print(f'prompt_new:{prompt_new}')
            # print(f'prompt_new.shape:{prompt_new.shape}')
            # print(f'torch.linalg.norm(prompt_new - prompt):{torch.linalg.norm(prompt_new - prompt)}')

            AE_loss = torch.pow(torch.linalg.norm(prompt_new - prompt)/prompt.shape[0], 2)
            # print(f'AE_loss:{AE_loss}')

            while AE_loss > 5:
                AE_loss = AE_loss / 10
            
            # print(f'Normalized AE_loss:{AE_loss}')
            # -----------------------------------------------------------------------------------------------


            '''
            Overall objective loss
            '''
            # removing trg ent
            # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
            # loss = CE_loss + AE_loss
            loss = CE_loss + AE_loss + loss_reconstruct
            # loss = CE_loss + loss_reconstruct

            


            loss.backward()
            optimizer.step()
            optimizer_AE.step()
            # self.decoder_optimizer.step()
            # self.global_step+=1

            # Model moment trainable parameters
            for name, param in self.modelmoment.named_parameters():
                if param.requires_grad:
                    print(name, param.data)


            # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
            # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
            losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
            # losses = {'entropy_loss': CE_loss.detach().item(), 'Total_loss': loss.detach().item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

            # if val_loss < self.best_val_loss:
            #     print("Epoch {} | val_loss improved from {:.6f} to {:.6f}".format(epoch, self.best_val_loss,
            #                                                                            val_loss))
            #     logging.info("Epoch {} | Val_loss improved from {:.6f} to {:.6f}".format(epoch, self.best_val_loss,
            #                                                                            val_loss))
            #     self.best_val_loss = val_loss
            #     self.best_model = deepcopy(modelmoment.state_dict())

                
            # self.optimizer.step()
            # self.optimizer2.step()
            # self.optimizer3.step()
            # self.optimizer4.step()            

            # self.optimizer2.step()
            # self.tov_optimizer.step()

            # losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
            # losses = {'entropy_loss': trg_ent.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
            # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
            # losses = {'entropy_loss': CE_loss.detach().item(), 'Total_loss': loss.detach().item()}
            # for key, val in losses.items():
            #     avg_meter[key].update(val, 32)

        return avg_meter['Total_loss']

class B2TSDA_CEOnly(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_CEOnly, self).__init__(configs)

        self.best_model_net1 = True
        self.pretrained_source = False

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "num_layer":24,
                "prompt_init": "uniform",
            },
        )
        self.sourceModel.init()

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "num_layer":24,
                "prompt_init": "uniform",
            },
        )
        self.modelmoment.init()

        self.modelmoment2 = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "num_layer":24,
                "prompt_init": "uniform",
            },
        )
        self.modelmoment2.init()

        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module
        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module
        self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        self.modelmoment2 = self.modelmoment2.module


        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.modelmoment2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.sourceModel.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()

                # forward pass correct sequences
                outputs = self.sourceModel(src_x)

                # classifier predictions
                src_pred = outputs.logits

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.sourceModel.state_dict())
        return src_only_model

    def update(self, trg_dataloader, val_dataloader, avg_meter, logger, source_model_dir):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = self.modelmoment.state_dict()
        self.last_model = self.modelmoment.state_dict()
        self.source_model_dir = source_model_dir

        print(f'Train CE Only')
        print(f'self.pretrained_source:{self.pretrained_source}')

        if not self.pretrained_source:
            print(f'Load pretrained source model..')
            load_source_model_path = source_model_dir + "/checkpoint.pt"
            self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
            for param in self.sourceModel.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')

        total_epochs = self.hparams["num_epochs"] 
        forget_rate = np.ones(total_epochs) * self.hparams["forget_rate"]
        forget_rate[:(self.hparams["warm_target"]+self.hparams["num_gradual"])] = np.linspace(0, self.hparams["forget_rate"], (self.hparams["warm_target"]+self.hparams["num_gradual"]))
        # print(f'forget_rate:{forget_rate}')

        # train
        for epoch in range(1, total_epochs+1):
        # for epoch in range(1, self.hparams["num_epochs"] + 1):
            if epoch <= round(total_epochs/2):#total_epochs
                alpha = 0.0
            else:
                alpha = (epoch*2-total_epochs)/total_epochs

            
            self.modelmoment.train()
            self.modelmoment2.train()

            Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            

            self.lr_scheduler.step()
            self.lr_scheduler2.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and (Total_loss_network.avg < best_src_risk or Total_loss_network2.avg < best_src_risk):
                # best_src_risk = avg_meter['Total_loss'].avg

                if Total_loss_network.avg < Total_loss_network2.avg:
                    best_src_risk = Total_loss_network.avg
                    self.best_model = deepcopy(self.modelmoment.state_dict())
                    self.best_model_net1 = True
                else:
                    best_src_risk = Total_loss_network2.avg
                    self.best_model = deepcopy(self.modelmoment2.state_dict())
                    self.best_model_net1 = False

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def target_train(self, trg_dataloader, val_dataloader, avg_meter, modelmoment, optimizer, modelmoment_ind, sourceModel, forget_rate, alpha, AE_cls, optimizer_AE, epoch, logging):

        for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

            # Target data
            trg_x = trg_x.float().to(self.device)

            # Validation data
            val_x, val_y, _ = next(iter(val_dataloader))
            val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)
            

            with torch.no_grad():
                outputs_src = self.sourceModel(trg_x)
                trg_pred_src = outputs_src.logits

                src_prob = nn.Softmax(dim=1)(trg_pred_src)
                _, pseudo_labels = torch.max(src_prob, 1)

            optimizer.zero_grad()

            # Co-training model output
            with torch.no_grad():
                outputs_ind = modelmoment_ind(trg_x)
                trg_pred_ind = outputs_ind.logits

            # Student's model output
            outputs = modelmoment(trg_x)
            trg_pred = outputs.logits

            # co-training loss
            loss_cot = F.cross_entropy(trg_pred_ind, pseudo_labels, reduction='none') # reduction='none' to get loss per batch instead of avg loss
            
            # get (1-R) percent low loss samples
            ind_loss_sorted = np.argsort(loss_cot.cpu().data).to(self.device)
            
            remember_rate = 1 - (1-alpha) * forget_rate
            num_remember = math.ceil(remember_rate * len(ind_loss_sorted))
            ind_loss_update = ind_loss_sorted[:num_remember]
            # ind_loss_neg_update = ind_loss_sorted[(num_remember):]#/*2+num_neg

            pseudo_labels_sorted = pseudo_labels[ind_loss_update].to(self.device)

            CE_loss = self.cross_entropy(trg_pred[ind_loss_update], pseudo_labels_sorted)

            '''
            Overall objective loss
            '''
            loss = CE_loss

            loss.backward()
            optimizer.step()

            losses = {'entropy_loss': CE_loss.detach().item(), 'Total_loss': loss.detach().item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        return avg_meter['Total_loss']


class B2TSDA_NoCOT_source(Algorithm):

    def __init__(self, configs, hparams, device):
        super(B2TSDA_NoCOT_source, self).__init__(configs)

        # device
        self.device = device
        self.hparams = hparams

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout_src,
            },
        )
        # self.sourceModel.to(self.device)
        self.sourceModel.init()
        # print(f'self.sourceModel.device:{self.sourceModel.device}')
        # for name, param in self.sourceModel.named_parameters():
        #             print(f"Parameter {name} is on device: {param.device}")

        self.pre_optimizer = torch.optim.Adam(
            self.sourceModel.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # for name, param in self.sourceModel.named_parameters():
        #             print(f"Parameter {name} is on device: {param.device}")

        # losses
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        # self.sourceModel.init()
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _, _, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                
                # print(f'self.sourceModel:{self.sourceModel}')
                # for name, param in self.sourceModel.named_parameters():
                #     print(f"Parameter {name} is on device: {param.device}")

                # for param in self.sourceModel.parameters():
                #     if not param.data.is_cuda:
                #         # print(f'model_t_all[0] param.data:{param.data}')
                #         # print(f'param.data.davice:{param.data.device}')
                #         param.data = param.to('cuda')

                self.pre_optimizer.zero_grad()
                # print(f'src_x:{src_x}')
                # print(f'src_x.device:{src_x.device}')

                # forward pass correct sequences
                outputs = self.sourceModel(src_x)
                # src_feat, _ = self.feature_extractor(src_x)


                # classifier predictions
                src_logits = outputs.logits
                # src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_logits, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.sourceModel.state_dict())
        return src_only_model

class B2TSDA_NoCOT_target(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_NoCOT_target, self).__init__(configs)

        # device
        self.device = device
        self.hparams = hparams
        self.m = hparams["m"]

        # self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        # self.AE_cls = nn.DataParallel(self.AE_cls)
        # self.AE_cls = self.AE_cls.module
        # self.AE_cls.to(self.device)

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout_src,
            },
        )
        # self.sourceModel.to(self.device)
        self.sourceModel.init()
        
        # print(f'self.sourceModel.device:{self.sourceModel.device}')
        # for name, param in self.sourceModel.named_parameters():
        #             print(f"Parameter {name} is on device: {param.device}")

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        # self.teacherModel = MOMENTPipeline.from_pretrained(
        #     "AutonLab/MOMENT-1-large", 
        #     model_kwargs={
        #         "task_name": "classification",
        #         "n_channels": self.configs.input_channels,
        #         "num_class": self.configs.num_classes,
        #         "sequence_len": self.configs.sequence_len,
        #         "num_layer": self.configs.prompt_length,
        #         "prompt_init": "uniform",
        #         "dropout": configs.dropout,
        #     },
        # )
        # self.teacherModel.init()

        # self.domain_classifier = DomainClassifier(feature_dim=configs.TSlength_aligned).to(self.device)



        # self.PreTrainedModel = MOMENTPipeline.from_pretrained(
        #     "AutonLab/MOMENT-1-large", 
        #     model_kwargs={
        #         "task_name": "classification",
        #         "n_channels": self.configs.input_channels,
        #         "num_class": self.configs.num_classes,
        #         "sequence_len": self.configs.sequence_len,
        #         "num_layer": self.configs.prompt_length,
        #         "prompt_init": "uniform",
        #         "dropout": configs.dropout,
        #         "freeze_head": True,
        #     },
        # )
        # self.PreTrainedModel.init()

        # print(f'self.sourceModel:{self.sourceModel}')

        # self.sourceModel = nn.DataParallel(self.sourceModel)
        # self.sourceModel = self.sourceModel.module
        # self.modelmoment = nn.DataParallel(self.modelmoment)
        # self.modelmoment = self.modelmoment.module
        # self.PreTrainedModel = nn.DataParallel(self.PreTrainedModel)
        # self.PreTrainedModel = self.PreTrainedModel.module
        # self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        # self.modelmoment2 = self.modelmoment2.module

        # self.AE_cls = nn.DataParallel(self.AE_cls)
        # self.AE_cls = self.AE_cls.module
        # self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        # self.AE_cls2 = self.AE_cls2.module

        
        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # self.optimizerAE = torch.optim.Adam(
        #     self.AE_cls.parameters(),
        #     lr=hparams["learning_rate_AE"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.domain_optimizer = torch.optim.Adam(self.domain_classifier.parameters(), lr=1e-4)

        # self.optimizerTeacher = torch.optim.Adam(
        #     self.teacherModel.parameters(),
        #     lr=hparams["learning_rate"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.optimizerRefine = torch.optim.Adam(
        #     self.modelmoment.parameters(),
        #     lr=hparams["learning_rate_refine"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.optimizer2 = torch.optim.Adam(
        #     self.modelmoment2.parameters(),
        #     lr=hparams["learning_rate"],
        #     weight_decay=hparams["weight_decay"]
        # )

        

        # self.optimizerAE2 = torch.optim.Adam(
        #     self.AE_cls2.parameters(),
        #     lr=hparams["learning_rate_AE"],
        #     weight_decay=hparams["weight_decay"]
        # )

        

        

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerTeacher = StepLR(self.optimizerTeacher, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerRefine = StepLR(self.optimizerRefine, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        
        # self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    # def update(self, trg_dataloader, avg_meter, logger, source_model_dir):
    # def update(self, trg_dataloader, avg_meter, logger):

    def update(self, trg_dataloader, trg_test_dataloader, avg_meter, logger, num_neighbors, source_model_dir):

        best_src_risk = float('inf')
        self.best_model = self.modelmoment.state_dict()
        self.last_model = self.modelmoment.state_dict()
        # self.teacherModel.load_state_dict(self.modelmoment.state_dict()) 

        load_source_model_path = source_model_dir + "/checkpoint.pt"
        self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
        self.sourceModel.eval()

        total_epochs = self.hparams["num_epochs"] 

        for epoch in range(0, total_epochs):
        
            gt_labels, pred_labels = [], []

            for step, (trg_x, trg_y, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)
                trg_y = trg_y.long().to(self.device)

                self.optimizer.zero_grad()
                # self.optimizerAE.zero_grad()

                # Source Model Predictions
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits
                    src_feat = outputs_src.embeddings

                    src_prob = F.softmax(trg_pred_src, dim=1)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)

                    # # Confidence-based pseudo-labeling (ignoring low-confidence samples)
                    # threshold = 0.85  # Tune this hyperparameter
                    # mask = src_conf > threshold
                    # pseudo_labels[~mask] = -1  # Ignore low-confidence predictions

                gt_labels.append(trg_y)
                pred_labels.append(pseudo_labels)

                # Target Model Output
                outputs = self.modelmoment(trg_x)
                trg_logits = outputs.logits
                trg_feat = outputs.embeddings
                trg_prob = F.softmax(trg_logits, dim=1)

                # ## EMA update of the Teacher Model
                # with torch.no_grad():                    
                #     self.teacher_model_update() 

                # Loss 1: Cross-Entropy with pseudo-labels
                # CE_loss = self.cross_entropy(trg_logits[mask], pseudo_labels[mask]) if mask.any() else torch.tensor(0.0).to(self.device)
                CE_loss = self.cross_entropy(trg_logits, pseudo_labels) 

                # # Loss 2: Prompt Reconstruction Loss (AutoEncoder)
                # AE_loss = 0
                # prompt = self.modelmoment.prompt.prompt
                # prompt_new = self.AE_cls(prompt)
                # AE_loss = torch.pow(torch.linalg.norm(prompt_new - prompt) / prompt.shape[0], 2)

                # AE_loss = AE_loss / 10 if AE_loss > 5 else AE_loss

                # # Loss 3: Feature Alignment (Cosine Similarity)
                # loss_alignment = torch.mean(1 - F.cosine_similarity(trg_feat, src_feat, dim=1))

                # # Loss 4: Consistency Loss (KL Divergence between Teacher & Student)
                # with torch.no_grad():
                #     teacher_logits = self.teacherModel(trg_x).logits
                # consistency_loss = self.kl_loss(F.log_softmax(trg_logits, dim=1), F.softmax(teacher_logits, dim=1))

                # # Loss 5: Domain-Invariant Learning (Domain Classifier)
                # # domain_logits = self.domain_classifier(trg_feat)  # Assume you define a small domain classifier
                # # domain_loss = F.cross_entropy(domain_logits, torch.ones_like(domain_logits))

                # domain_logits = self.domain_classifier(trg_feat)  # Predict domain
                # domain_labels = torch.ones_like(domain_logits)  # 1 = target domain
                # domain_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)


                # # Loss 6: Input Reconstruction Loss
                # mt = torch.bernoulli(torch.empty_like(trg_x).uniform_(0, 1)).to(self.device)
                # m_ones = torch.ones_like(trg_x).to(self.device)

                # sum_mt = mt.flatten().sum()
                # c_mask = sum_mt / trg_x.numel()
                # c_unmask = (trg_x.numel() - sum_mt) / trg_x.numel()

                # src2 = trg_x * (m_ones - mt)
                # gamma = 0.5
                # criterion = nn.MSELoss()

                # out = self.modelmoment.forward_reconstruct(src2)
                # pred_reconstruct = out.reconstruction

                # weight_masked = c_mask / (c_mask + c_unmask)
                # weight_unmasked = c_unmask / (c_mask + c_unmask)
                # loss_reconstruct = weight_masked * criterion(pred_reconstruct, src2) + weight_unmasked * criterion(pred_reconstruct, src2)

                # # === Added Contrastive Learning ===
                # # Compute outputs for the weak and strong augmented views
                # outputs_weak = self.modelmoment(trg_x_weak)
                # outputs_strong = self.modelmoment(trg_x_strong)
                # feat_weak = outputs_weak.embeddings
                # feat_strong = outputs_strong.embeddings

                # # Compute the contrastive loss using a helper function
                # contrastive_loss_value = self.contrastive_loss(feat_weak, feat_strong, temperature=self.hparams.get("temperature", 0.5))
                # # contrastive_weight = self.hparams.get("contrastive_weight", 0.1)

                # Final Loss Combination (added contrastive loss term)
                loss = (
                    CE_loss
                    # + 0.0005 * AE_loss  # Reduce AE loss impact further
                    # + loss_reconstruct
                    # + 0.1 * consistency_loss
                    # + 0.1 * loss_alignment
                    # + 0.1 * domain_loss
                    # + 0.1 * contrastive_loss_value
                )

                loss.backward()
                self.optimizer.step()
                # self.optimizerAE.step()
                # self.domain_optimizer.step()  # Update the domain classifier

                losses = {
                    'entropy_loss': CE_loss.detach().item(),
                    # 'AE_loss': AE_loss.detach().item(),
                    # 'Loss_reconstruct': loss_reconstruct.detach().item(),
                    # 'Consistency_loss': consistency_loss.detach().item(),
                    # 'Loss_alignment': loss_alignment.detach().item(),
                    # 'Domain_loss': domain_loss.detach().item(),
                    # 'Contrastive_loss': contrastive_loss_value.detach().item(),
                    'Total_loss': loss.detach().item()
                }

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()
            # self.lr_schedulerAE.step()

            gt_labels = torch.cat(gt_labels).to('cpu')
            pred_labels = torch.cat(pred_labels).to('cpu')
            acc = 100. * accuracy_score(gt_labels, pred_labels)
            print(f'Pseudo-acc: {acc:.2f}%')

            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def teacher_model_update(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.modelmoment.parameters(), self.teacherModel.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def contrastive_loss(self, embedding1, embedding2, temperature=0.5):
        """
        Compute a simple NT-Xent (InfoNCE) contrastive loss between two batches of embeddings.
        Here, for a batch of size N, we assume that for each sample i in embedding1,
        the corresponding sample i in embedding2 forms the positive pair.
        """
        # If embeddings are 3D, average over the sequence dimension
        if embedding1.dim() == 3:
            embedding1 = embedding1.mean(dim=1)
        if embedding2.dim() == 3:
            embedding2 = embedding2.mean(dim=1)

        batch_size = embedding1.size(0)
        # Normalize the embeddings
        z1 = F.normalize(embedding1, dim=1)
        z2 = F.normalize(embedding2, dim=1)
        # Compute cosine similarity matrix [N x N]
        similarity_matrix = torch.mm(z1, z2.t())
        logits = similarity_matrix / temperature
        labels = torch.arange(batch_size).to(embedding1.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return (loss_a + loss_b) / 2


class B2TSDA_COT(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_COT, self).__init__(configs)

        self.best_model_net1 = True
        self.pretrained_source = False

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        self.modelmoment2 = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment2.init()

        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module
        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module
        self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        self.modelmoment2 = self.modelmoment2.module


        self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls = nn.DataParallel(self.AE_cls)
        self.AE_cls = self.AE_cls.module

        self.AE_cls2 = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        self.AE_cls2 = self.AE_cls2.module



        self.AE_cls = nn.DataParallel(self.AE_cls)
        self.AE_cls = self.AE_cls.module
        self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        self.AE_cls2 = self.AE_cls2.module


        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.modelmoment2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE = torch.optim.Adam(
            self.AE_cls.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE2 = torch.optim.Adam(
            self.AE_cls2.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.sourceModel.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()

                # forward pass correct sequences
                outputs = self.sourceModel(src_x)

                # classifier predictions
                src_pred = outputs.logits

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.sourceModel.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger, source_model_dir):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = self.modelmoment.state_dict()
        self.last_model = self.modelmoment.state_dict()
        self.source_model_dir = source_model_dir

        if not self.pretrained_source:
            print(f'Load pretrained source model..')
            load_source_model_path = source_model_dir + "/checkpoint.pt"
            print(f'source model path:{load_source_model_path}')
            self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
            for param in self.sourceModel.parameters():
                if not param.data.is_cuda:
                    param.data = param.to('cuda')

        total_epochs = self.hparams["num_epochs"] 
        
        rate_schedule = np.ones(total_epochs) * self.hparams["forget_rate"]
        rate_schedule[:(self.hparams["num_gradual"])] = np.linspace(0, self.hparams["forget_rate"]**self.hparams["exponent"], self.hparams["num_gradual"])
        print(f'rate_schedule:{rate_schedule}')

        # train
        for epoch in range(0, total_epochs):
            
            self.modelmoment.train()
            self.modelmoment2.train()

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)

                # Co-teaching model output
                with torch.no_grad():
                    outputs_ind = self.modelmoment2(trg_x)
                    trg_pred_ind = outputs_ind.logits
                    trg_ind_prob = nn.Softmax(dim=1)(trg_pred_ind)

                # Target Model1 output
                outputs = self.modelmoment(trg_x)
                trg_pred = outputs.logits

                # Target Model2 output
                outputs2 = self.modelmoment2(trg_x)
                trg_pred2 = outputs2.logits

                # co-teaching loss
                loss_1, loss_2 = self.loss_coteaching(trg_pred, trg_pred2, pseudo_labels, rate_schedule[epoch], trg_idx)

                # reconstruct -------------------------------------------------------------
                mt = random.uniform(0,1) #mask
                s0,s1,s2 = trg_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                mt = torch.bernoulli(randuniform).to(self.device)
                m_ones = torch.ones(s0,s1,s2).to(self.device)

                sum_mt = mt.flatten().sum()
                c_mask = sum_mt/(s0*s1*s2)
                c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)

                #src if mt=1 -> x=0
                src2 = torch.clone(trg_x)
                src2 = src2 * (m_ones-mt)
                gamma = 0.5
                criterion = RMSELoss()

                # pred reconstruct
                out = self.modelmoment.forward_reconstruct(src2)
                pred_reconstruct = out.reconstruction

                out2 = self.modelmoment2.forward_reconstruct(src2)
                pred_reconstruct2 = out2.reconstruction
                
                #loss reconstruct
                loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)
                loss_reconstruct2 = gamma * c_mask * criterion(pred_reconstruct2, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct2, src2)
        
                # -------------------------------------------------------------------------


                # prompt reconstruction -------------------------------------------------------------------------
                AE_loss = 0
                AE_loss2 = 0
                # use prompt
                prompt = self.modelmoment.prompt.prompt
                prompt_new = self.AE_cls(prompt)
                prompt2 = self.modelmoment2.prompt.prompt
                prompt_new2 = self.AE_cls2(prompt2)

                AE_loss = torch.pow(torch.linalg.norm(prompt_new - prompt)/prompt.shape[0], 2)
                AE_loss2 = torch.pow(torch.linalg.norm(prompt_new2 - prompt2)/prompt2.shape[0], 2)

                while AE_loss > 5:
                    AE_loss = AE_loss / 10

                while AE_loss2 > 5:
                    AE_loss2 = AE_loss2 / 10
                
                # -----------------------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                loss = loss_1 + AE_loss + loss_reconstruct
                loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.optimizer2.zero_grad()
                loss2.backward()
                self.optimizer2.step()

                self.optimizerAE.step()
                self.optimizerAE2.step()

                losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                 'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)


            self.lr_scheduler.step()
            self.lr_scheduler2.step()
            self.lr_schedulerAE.step()
            self.lr_schedulerAE2.step()


            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def loss_coteaching(self, y_1, y_2, t, forget_rate, ind):
        loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
        ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
        ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

    def target_train(self, trg_dataloader, val_dataloader, avg_meter, modelmoment, optimizer, modelmoment_ind, sourceModel, forget_rate, alpha, AE_cls, optimizer_AE, epoch, logging):

        for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

            # Target data
            trg_x = trg_x.float().to(self.device)

            # Validation data
            val_x, val_y, _ = next(iter(val_dataloader))
            val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)
            

            with torch.no_grad():
                outputs_src = self.sourceModel(trg_x)
                trg_pred_src = outputs_src.logits

                src_prob = nn.Softmax(dim=1)(trg_pred_src)
                src_conf, pseudo_labels = torch.max(src_prob, 1)

            optimizer.zero_grad()
            optimizer_AE.zero_grad()

            # Co-training model output
            with torch.no_grad():
                outputs_ind = modelmoment_ind(trg_x)
                trg_pred_ind = outputs_ind.logits
                trg_ind_prob = nn.Softmax(dim=1)(trg_pred_ind)

            # Student's model output
            outputs = modelmoment(trg_x)
            trg_pred = outputs.logits

            # reconstruct -------------------------------------------------------------
            mt = random.uniform(0,1) #mask
            s0,s1,s2 = trg_x.shape
            randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
            mt = torch.bernoulli(randuniform).to(self.device)
            m_ones = torch.ones(s0,s1,s2).to(self.device)

            sum_mt = mt.flatten().sum()
            c_mask = sum_mt/(s0*s1*s2)
            c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)

            #src if mt=1 -> x=0
            src2 = torch.clone(trg_x)
            src2 = src2 * (m_ones-mt)
            gamma = 0.5
            criterion = RMSELoss()

            # pred reconstruct
            out = self.modelmoment.forward_reconstruct(src2)
            pred_reconstruct = out.reconstruction
            
            #loss reconstruct
            loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)
            
            # --------------------------------------------------------------------------

            # co-training loss
            loss_cot = F.cross_entropy(trg_pred_ind, pseudo_labels, reduction='none') # reduction='none' to get loss per batch instead of avg loss
            
            # get (1-R) percent low loss samples
            ind_loss_sorted = np.argsort(loss_cot.cpu().data).to(self.device)
            
            remember_rate = 1 - (1-alpha) * forget_rate
            num_remember = math.ceil(remember_rate * len(ind_loss_sorted))
            ind_loss_update = ind_loss_sorted[:num_remember]

            pseudo_labels_sorted = pseudo_labels[ind_loss_update].to(self.device)

            CE_loss = self.cross_entropy(trg_pred[ind_loss_update], pseudo_labels_sorted)

            # use prompt
            prompt = modelmoment.prompt.prompt

            # prompt reconstruction -------------------------------------------------------------------------
            AE_loss = 0
            prompt_new = AE_cls(prompt)

            AE_loss = torch.pow(torch.linalg.norm(prompt_new - prompt)/prompt.shape[0], 2)

            while AE_loss > 5:
                AE_loss = AE_loss / 10
            
            # -----------------------------------------------------------------------------------------------


            '''
            Overall objective loss
            '''
            loss = CE_loss + AE_loss + loss_reconstruct


            loss.backward()
            optimizer.step()
            optimizer_AE.step()
            
            losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

            

        return avg_meter['Total_loss']

class B2TSDA_COT_source(Algorithm):

    def __init__(self, configs, hparams, device):
        super(B2TSDA_COT_source, self).__init__(configs)

        
        self.best_model_net1 = True
        self.pretrained_source = False
        self.f1 = F1Score(task="multiclass", num_classes=configs.num_classes, average="macro").to(device)

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()


        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module

        self.pre_optimizer = torch.optim.Adam(
            self.sourceModel.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # device
        self.device = device
        self.hparams = hparams

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, src_dataloader_test, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _, _, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()

                # forward pass correct sequences
                outputs = self.sourceModel(src_x)

                # classifier predictions
                src_pred = outputs.logits

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
            
        src_only_model = deepcopy(self.sourceModel.state_dict())
        return src_only_model

    def eval(self, test_loader):

        model = self.sourceModel
        model.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                outputs = model(data)
                # predictions = model(data)
                predictions = outputs.logits

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.full_preds_eval = torch.cat((preds_list))
        self.full_labels_eval = torch.cat((labels_list))

    def calc_metrics(self, test_loader, logger):
       
        self.eval(test_loader)
        
        f1 = self.f1(self.full_preds_eval.argmax(dim=1), self.full_labels_eval).item()
        
        logger.debug(f'f1\t:{f1}')

# Ema pseudo label
class B2TSDA_COT_target(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_COT_target, self).__init__(configs)

        self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls = nn.DataParallel(self.AE_cls)
        self.AE_cls = self.AE_cls.module

        self.AE_cls2 = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        self.AE_cls2 = self.AE_cls2.module

        self.best_model_net1 = True
        self.pretrained_source = False

        self.acc = Accuracy(task="multiclass", num_classes=configs.num_classes).to(device)
        self.f1 = F1Score(task="multiclass", num_classes=configs.num_classes, average="macro").to(device)
        self.auroc = AUROC(task="multiclass", num_classes=configs.num_classes).to(device)  

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        self.modelmoment2 = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment2.init()

        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module
        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module
        self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        self.modelmoment2 = self.modelmoment2.module

        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.modelmoment2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE = torch.optim.Adam(
            self.AE_cls.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE2 = torch.optim.Adam(
            self.AE_cls2.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        # device and hyperparameters
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        # --- Self-distillation teacher buffers for two target models ---
        # They will be initialized on the first call to update()
        self.ps_buffer1 = None  # for self.modelmoment
        self.ps_buffer2 = None  # for self.modelmoment2

    def update(self, trg_dataloader, trg_dataloader_test, avg_meter, logger, source_model_dir):

        # Defining best and last model
        best_src_risk = float('inf')
        self.model1 = self.modelmoment.state_dict()  # COT
        self.model2 = self.modelmoment2.state_dict()
        self.source_model_dir = source_model_dir

        if not self.pretrained_source:
            print('Load pretrained source model..')
            load_source_model_path = source_model_dir + "/checkpoint.pt"
            print(f'source model path: {load_source_model_path}')
            self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
            for param in self.sourceModel.parameters():
                if not param.data.is_cuda:
                    param.data = param.data.to('cuda')
            self.sourceModel.eval()
            for param in self.sourceModel.parameters():
                param.requires_grad = False

        total_epochs = self.hparams["num_epochs"]
        kd_weight = 1

        # Initialize teacher buffers if not already done.
        num_samples = len(trg_dataloader.dataset)
        if self.ps_buffer1 is None:
            self.ps_buffer1 = torch.zeros(num_samples, self.configs.num_classes).to(self.device)
        if self.ps_buffer2 is None:
            self.ps_buffer2 = torch.zeros(num_samples, self.configs.num_classes).to(self.device)

        gma = self.hparams["gma"]

        for epoch in range(0, total_epochs):

            self.modelmoment.train()
            self.modelmoment2.train()

            for step, (trg_x, _, trg_idx, _, _) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)

                # -----------------------------
                # Obtain source model soft-labels and pseudo-labels.
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits
                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                # -----------------------------

                # Target model outputs from both branches.
                outputs = self.modelmoment(trg_x)
                trg_pred = outputs.logits

                outputs2 = self.modelmoment2(trg_x)
                trg_pred2 = outputs2.logits

                # --- Self-distillation: update teacher predictions for both target models ---
                if epoch == 0:
                    # Initialization using Equation (8):
                    # For self.modelmoment (buffer1)
                    teacher_preds1 = src_prob.clone()
                    top_val, top_idx = teacher_preds1.max(dim=1, keepdim=True)
                    uniform_rest = (1 - top_val) / (self.configs.num_classes - 1)
                    teacher_preds1 = uniform_rest.expand_as(teacher_preds1).clone()
                    teacher_preds1.scatter_(1, top_idx, top_val)
                    self.ps_buffer1[trg_idx] = teacher_preds1.detach()
                    # For self.modelmoment2 (buffer2)
                    teacher_preds2 = src_prob.clone()
                    top_val2, top_idx2 = teacher_preds2.max(dim=1, keepdim=True)
                    uniform_rest2 = (1 - top_val2) / (self.configs.num_classes - 1)
                    teacher_preds2 = uniform_rest2.expand_as(teacher_preds2).clone()
                    teacher_preds2.scatter_(1, top_idx2, top_val2)
                    self.ps_buffer2[trg_idx] = teacher_preds2.detach()
                else:
                    with torch.no_grad():
                        # For self.modelmoment (buffer1)
                        old_teacher_preds1 = self.ps_buffer1[trg_idx]
                        
                        current_preds1 = nn.Softmax(dim=1)(trg_pred)
                        
                        teacher_preds1 = gma * old_teacher_preds1 + (1-gma) * current_preds1
                        self.ps_buffer1[trg_idx] = teacher_preds1.detach()

                        # For self.modelmoment2 (buffer2)
                        old_teacher_preds2 = self.ps_buffer2[trg_idx]
                        
                        current_preds2 = nn.Softmax(dim=1)(trg_pred2)
                        teacher_preds2 = gma * old_teacher_preds2 + (1-gma) * current_preds2
                        self.ps_buffer2[trg_idx] = teacher_preds2.detach()
                # --- End self-distillation update ---

                src_conf1, pseudo_labels1 = torch.max(teacher_preds1, 1)
                src_conf2, pseudo_labels2 = torch.max(teacher_preds2, 1)

                # -----------------------------
                # Input Reconstruction branch.
                s0, s1, s2 = trg_x.shape
                randuniform = torch.empty(s0, s1, s2).uniform_(0, 1)
                mt = torch.bernoulli(randuniform).to(self.device)
                total_elements = s0 * s1 * s2
                c_mask = mt.sum() / total_elements
                c_unmask = (total_elements - mt.sum()) / total_elements

                src2 = trg_x * (1 - mt)

                gamma = 0.5
                criterion = RMSELoss()
                out = self.modelmoment.forward_reconstruct(src2)
                pred_reconstruct = out.reconstruction

                out2 = self.modelmoment2.forward_reconstruct(src2)
                pred_reconstruct2 = out2.reconstruction

                loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1 - gamma) * c_unmask * criterion(pred_reconstruct, src2)
                loss_reconstruct2 = gamma * c_mask * criterion(pred_reconstruct2, src2) + (1 - gamma) * c_unmask * criterion(pred_reconstruct2, src2)
                # -----------------------------

                # Prompt reconstruction branch. ---------------
                prompt = self.modelmoment.prompt.prompt
                prompt_new = self.AE_cls(prompt)
                prompt2 = self.modelmoment2.prompt.prompt
                prompt_new2 = self.AE_cls2(prompt2)

                AE_loss = 0.0005 * torch.pow(torch.linalg.norm(prompt_new - prompt) / prompt.shape[0], 2)
                AE_loss2 = 0.0005 * torch.pow(torch.linalg.norm(prompt_new2 - prompt2) / prompt2.shape[0], 2)
                # ---------------------------------------------

                # --- Knowledge Distillation (KD) losses ---
                # KLDivLoss expects the input to be log-probabilities.
                loss_kl1 = self.kl_loss(torch.log_softmax(trg_pred, dim=1), teacher_preds1)
                loss_kl2 = self.kl_loss(torch.log_softmax(trg_pred2, dim=1), teacher_preds2)
                # --- End KD losses ---

                # Total loss
                loss = kd_weight * loss_kl1 + AE_loss + loss_reconstruct
                loss2 = kd_weight * loss_kl2 + AE_loss2 + loss_reconstruct2 


                for opt in (self.optimizer, self.optimizer2, self.optimizerAE, self.optimizerAE2):
                    opt.zero_grad()

                (loss + loss2).backward()   # pytorch sums the grads across the two branches 
                                             # if they share any parameters

                for opt in (self.optimizer, self.optimizer2, self.optimizerAE, self.optimizerAE2):
                    opt.step()

                losses = {
                    
                    'KL_loss1': loss_kl1.detach().item(),
                    'AE_loss': AE_loss.detach().item(),
                    'Loss_reconstruct': loss_reconstruct.detach().item(),
                    'Total_loss': loss.detach().item(),
                    
                    'KL_loss2': loss_kl2.detach().item(),
                    'AE_loss2': AE_loss2.detach().item(),
                    'Loss_reconstruct2': loss_reconstruct2.detach().item(),
                    'Total_loss2': loss2.detach().item()
                }
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()
            self.lr_scheduler2.step()
            self.lr_schedulerAE.step()
            self.lr_schedulerAE2.step()
            torch.cuda.empty_cache()

            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug('-------------------------------------')
            self.calc_metrics(trg_dataloader_test, logger)

        self.model1 = deepcopy(self.modelmoment.state_dict())
        self.model2 = deepcopy(self.modelmoment2.state_dict())



        return self.model1, self.model2

    

    def loss_coteaching3(self, y_1, y_2, t):
        num_batch = y_1.shape[0]
        loss_1 = F.cross_entropy(y_1, t)
        loss_2 = F.cross_entropy(y_2, t)
        return torch.sum(loss_1) / num_batch, torch.sum(loss_2) / num_batch

    def loss_coteaching4(self, y_1, y_2, t1, t2):
        num_batch = y_1.shape[0]
        loss_1 = F.cross_entropy(y_1, t1)
        loss_2 = F.cross_entropy(y_2, t2)
        return torch.sum(loss_1) / num_batch, torch.sum(loss_2) / num_batch

    def eval_aggr(self, test_loader):

        model = self.modelmoment
        model2 = self.modelmoment2
        model.eval()
        model2.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                outputs = model(data)
                predictions = outputs.logits
                outputs2 = model2(data)
                predictions2 = outputs2.logits

                max_output, _ = torch.max(predictions, 1)
                max_output2, _ = torch.max(predictions2, 1)

                a = max_output/(max_output+max_output2)
                b = max_output2/(max_output+max_output2)

                # Aggregate output 
                output_target = torch.unsqueeze(a,1)*predictions + torch.unsqueeze(b,1)*predictions2

                # compute loss
                loss = F.cross_entropy(output_target, labels)
                total_loss.append(loss.item())
                pred = output_target.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss_eval = torch.tensor(total_loss).mean()  # average loss
        self.full_preds_eval = torch.cat((preds_list))
        self.full_labels_eval = torch.cat((labels_list))

    def calc_metrics(self, test_loader, logger):
       
        self.eval_aggr(test_loader)
        
        # accuracy  
        acc = self.acc(self.full_preds_eval.argmax(dim=1), self.full_labels_eval).item()
        # f1
        f1 = self.f1(self.full_preds_eval.argmax(dim=1), self.full_labels_eval).item()
        # auroc 
        auroc = self.auroc(self.full_preds_eval, self.full_labels_eval).item()

        logger.debug(f'f1\t:{f1}')
        

        return acc, f1, auroc



class B2TSDA_COT_source_multi(Algorithm):

    def __init__(self, configs, hparams, device):
        super(B2TSDA_COT_source, self).__init__(configs)

        self.best_model_net1 = True
        self.pretrained_source = False
        self.f1 = F1Score(task="multiclass", num_classes=configs.num_classes, average="macro").to(device)

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        self.sourceModel.to(device)

        self.pre_optimizer = torch.optim.Adam(
            self.sourceModel.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # device
        self.device = device
        self.hparams = hparams

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, src_dataloader_test, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _, _, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()

                # forward pass correct sequences
                outputs = self.sourceModel(src_x)

                # classifier predictions
                src_pred = outputs.logits

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
            self.calc_metrics(src_dataloader_test, logger)
        src_only_model = deepcopy(self.sourceModel.state_dict())
        return src_only_model

    def eval(self, test_loader):

        model = self.sourceModel
        model.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                outputs = model(data)
                # predictions = model(data)
                predictions = outputs.logits

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        # self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds_eval = torch.cat((preds_list))
        self.full_labels_eval = torch.cat((labels_list))

    def calc_metrics(self, test_loader, logger):
       
        self.eval(test_loader)
        
        
        # f1
        f1 = self.f1(self.full_preds_eval.argmax(dim=1), self.full_labels_eval).item()
        
        logger.debug(f'f1\t:{f1}')

class B2TSDA_COT_target_multi(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_COT_target_multi, self).__init__(configs)

        self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls = nn.DataParallel(self.AE_cls)
        self.AE_cls = self.AE_cls.module

        self.AE_cls2 = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        self.AE_cls2 = self.AE_cls2.module

        self.best_model_net1 = True
        self.pretrained_source = False

        self.acc = Accuracy(task="multiclass", num_classes=configs.num_classes).to(device)
        self.f1 = F1Score(task="multiclass", num_classes=configs.num_classes, average="macro").to(device)
        self.auroc = AUROC(task="multiclass", num_classes=configs.num_classes).to(device)  

        self.sourceModel = []
        for s in configs.src_domains:
            srcModel = MOMENTPipeline.from_pretrained(
                "AutonLab/MOMENT-1-large",
                model_kwargs={
                    "task_name": "classification",
                    "n_channels": self.configs.input_channels,
                    "num_class": self.configs.num_classes,
                    "sequence_len": self.configs.sequence_len,
                    "num_layer": self.configs.prompt_length,
                    "prompt_init": "uniform",
                    "dropout": configs.dropout,
                },
            )
            srcModel.init()
            srcModel = srcModel.to(device)
            self.sourceModel.append(srcModel)

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()
        self.modelmoment.to(device)

        self.modelmoment2 = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment2.init()
        self.modelmoment2.to(device)

        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.modelmoment2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE = torch.optim.Adam(
            self.AE_cls.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE2 = torch.optim.Adam(
            self.AE_cls2.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        # device and hyperparameters
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        # --- Self-distillation teacher buffers for two target models ---
        # They will be initialized on the first call to update()
        self.ps_buffer1 = None  # for self.modelmoment
        self.ps_buffer2 = None  # for self.modelmoment2
        M = len(configs.src_domains)
        # Start with uniform weights 1/M
        self.lambdas = [1.0 / M] * M
        self.alpha = hparams.get("alpha", 0.9)

    def update(self, trg_dataloader, trg_dataloader_test, avg_meter, logger, source_model_dir_list):
        # Defining best and last model
        best_src_risk = float('inf')
        self.model1 = self.modelmoment.state_dict()  # COT
        self.model2 = self.modelmoment2.state_dict()
        self.source_model_dir = source_model_dir_list
        M = len(self.sourceModel)


        if not self.pretrained_source:
            print('Load pretrained source model..')
            for i, src_dir in enumerate(source_model_dir_list):
                load_source_model_path = src_dir.rstrip('/') + "/checkpoint.pt"
                print(f'source model path: {load_source_model_path}')
                self.sourceModel[i].load_state_dict(torch.load(load_source_model_path)["non_adapted"])
                for param in self.sourceModel[i].parameters():
                    if not param.data.is_cuda:
                        param.data = param.data.to(self.device)
                self.sourceModel[i].eval()
                for param in self.sourceModel[i].parameters():
                    param.requires_grad = False
            # so we don't reload next time:
            self.pretrained_source = True

        total_epochs = self.hparams["num_epochs"]
        kd_weight    = 1
        gma          = self.hparams["gma"]

        num_samples = len(trg_dataloader.dataset)
        N_T = num_samples

        

        #  Initialize or load ps_buffer1/ps_buffer2 if needed (on CPU)
        if self.ps_buffer1 is None:
            self.ps_buffer1 = torch.zeros(num_samples, self.configs.num_classes, dtype=torch.float32)
        if self.ps_buffer2 is None:
            self.ps_buffer2 = torch.zeros(num_samples, self.configs.num_classes, dtype=torch.float32)

        for epoch in range(total_epochs):
            self.modelmoment.train()
            self.modelmoment2.train()

            # Compute dynamic alpha = N_p / N_T according to Eq (16)
            with torch.no_grad():
                # get per-sample confidences from the previous teacher buffer
                confidences, _ = self.ps_buffer1.max(dim=1)     # shape (N_T,)
            tau = self.hparams.get("pseudo_threshold", 0.0)    # threshold τ
            N_p = int((confidences > tau).sum().item())       # count samples above τ
            # avoid zero-division
            self.alpha = N_p / N_T if N_T > 0 else 0.0
            # ─────────────

            # 1) Compute per-source entropies & update lambdas
            source_entropies = []
            with torch.no_grad():
                for src_mod in self.sourceModel:
                    probs = []
                    for batch_x, _, idx, *_ in trg_dataloader:
                        out = src_mod(batch_x.to(self.device))
                        p = torch.softmax(out.logits, dim=1)
                        probs.append(p)
                    all_p = torch.cat(probs, dim=0)
                    H = -(all_p * torch.log(all_p + 1e-8)).sum(dim=1).mean().item()
                    source_entropies.append(H)

            etas = [1.0/(H + 1e-8) for H in source_entropies]
            max_eta = max(etas)
            new_lambdas = [eta / max_eta for eta in etas]
            for i in range(M):
                self.lambdas[i] = self.alpha * self.lambdas[i] + (1 - self.alpha) * new_lambdas[i]

            # Precompute weighted source probs buffer with updated lambdas
            self.src_prob_all = torch.zeros(num_samples, self.configs.num_classes)
            with torch.no_grad():
                for x, _, idx, _, _ in trg_dataloader:
                    x = x.float().to(self.device)
                    weighted = []
                    for i, src_mod in enumerate(self.sourceModel):
                        prob = torch.softmax(src_mod(x).logits, dim=1)
                        weighted.append(prob * self.lambdas[i])
                    src_batch = torch.stack(weighted, dim=0).sum(dim=0)
                    self.src_prob_all[idx] = src_batch.cpu()

            for step, (trg_x, _, trg_idx, _, _) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                if torch.isnan(trg_x).any():
                    raise RuntimeError(f"NaN in trg_x at epoch={epoch}, step={step}")
                B = trg_x.size(0)

                # ————————————
                #  Grab the precomputed source soft‐labels for these indices:
                src_prob_batch = self.src_prob_all[trg_idx].to(self.device)  # shape (B, C)
                if torch.isnan(src_prob_batch).any():
                    raise RuntimeError(f"NaN in src_prob_batch at epoch={epoch}, step={step}, idx={trg_idx}")

                # Compute “pseudo_labels” 
                if epoch == 0:
                    top_conf, top_idx = src_prob_batch.max(dim=1, keepdim=True)
                    if torch.isnan(top_conf).any() or torch.isnan(top_idx).any():
                        raise RuntimeError(f"NaN in top_conf or top_idx at epoch 0, step {step}")
                    # Build a “one‐hot”‐like distribution where top_idx entry = top_conf, all others = (1−top_conf)/(C−1)
                    uniform_rest = (1 - top_conf) / (self.configs.num_classes - 1)
                    if torch.isnan(uniform_rest).any():
                        raise RuntimeError(f"NaN in uniform_rest (division by zero?) at epoch 0, step {step}")
                    teacher1 = uniform_rest.expand_as(src_prob_batch).clone()
                    teacher1.scatter_(1, top_idx, top_conf)
                    if torch.isnan(teacher1).any():
                        raise RuntimeError(f"NaN in initial teacher1 at epoch 0, step {step}")
                    self.ps_buffer1[trg_idx] = teacher1.cpu()

                    teacher2 = uniform_rest.expand_as(src_prob_batch).clone()
                    teacher2.scatter_(1, top_idx, top_conf)
                    if torch.isnan(teacher2).any():
                        raise RuntimeError(f"NaN in initial teacher2 at epoch 0, step {step}")
                    self.ps_buffer2[trg_idx] = teacher2.cpu()
                else:
                    
                    old_teacher1 = self.ps_buffer1[trg_idx].to(self.device)
                    old_teacher2 = self.ps_buffer2[trg_idx].to(self.device)

                    if torch.isnan(old_teacher1).any():
                        raise RuntimeError(f"old_teacher1 is NaN at epoch {epoch}, step {step}")
                    if torch.isnan(old_teacher2).any():
                        raise RuntimeError(f"old_teacher2 is NaN at epoch {epoch}, step {step}")

                # ————————————
                # Run the two target branches + AE + KD exactly as before, but never re‐call sourceModel
                #     (instead, use “src_prob_batch” or the existing ps_buffer1/2).
                
                with torch.cuda.amp.autocast():
                    # Forward both target models once to get logits:
                    outputs1 = self.modelmoment(trg_x)    
                    logits1  = outputs1.logits
                    if torch.isnan(logits1).any():
                        raise RuntimeError(f"NaN in logits1 at epoch {epoch}, step {step}")
                    outputs2 = self.modelmoment2(trg_x)     
                    logits2  = outputs2.logits
                    if torch.isnan(logits2).any():
                        raise RuntimeError(f"NaN in logits2 at epoch {epoch}, step {step}")

                    prob1 = torch.softmax(logits1, dim=1)
                    prob2 = torch.softmax(logits2, dim=1)
                    if torch.isnan(prob1).any():
                        raise RuntimeError(f"NaN in prob1 (softmax) at epoch {epoch}, step {step}")
                    if torch.isnan(prob2).any():
                        raise RuntimeError(f"NaN in prob2 (softmax) at epoch {epoch}, step {step}")

                    if epoch > 0:
                        # Update teacher via self‐distillation:
                        teacher1 = gma * old_teacher1 + (1 - gma) * prob1
                        teacher2 = gma * old_teacher2 + (1 - gma) * prob2

                        # **Clamp & re‐normalize** so teacher never goes outside [0,1] or sums to zero:
                        teacher1 = torch.clamp(teacher1, min=1e-8, max=1.0)
                        teacher1 = teacher1 / teacher1.sum(dim=1, keepdim=True)
                        teacher2 = torch.clamp(teacher2, min=1e-8, max=1.0)
                        teacher2 = teacher2 / teacher2.sum(dim=1, keepdim=True)

                        if torch.isnan(teacher1).any():
                            raise RuntimeError(f"NaN in teacher1 after upd at epoch {epoch}, step {step}")
                        if torch.isnan(teacher2).any():
                            raise RuntimeError(f"NaN in teacher2 after upd at epoch {epoch}, step {step}")

                        self.ps_buffer1[trg_idx] = teacher1.detach().cpu()
                        self.ps_buffer2[trg_idx] = teacher2.detach().cpu()
                    else:
                        
                        teacher1 = src_prob_batch
                        teacher2 = src_prob_batch

                    # KL losses:
                    loss_kl1 = self.kl_loss(torch.log_softmax(logits1, dim=1),
                                            teacher1.to(self.device))
                    loss_kl2 = self.kl_loss(torch.log_softmax(logits2, dim=1),
                                            teacher2.to(self.device))

                    if torch.isnan(loss_kl1).any():
                        raise RuntimeError(f"NaN in loss_kl1 at epoch {epoch}, step {step}")
                    if torch.isnan(loss_kl2).any():
                        raise RuntimeError(f"NaN in loss_kl2 at epoch {epoch}, step {step}")

                    # AE prompt losses 
                    prompt      = self.modelmoment.prompt.prompt
                    if torch.isnan(prompt).any():
                        raise RuntimeError(f"NaN in prompt at epoch {epoch}, step {step}")
                    prompt_new  = self.AE_cls(prompt)
                    AE_loss1    = 0.0005 * (torch.norm(prompt_new - prompt) / prompt.shape[0])**2
                    if torch.isnan(AE_loss1).any():
                        raise RuntimeError(f"NaN in AE_loss1 at epoch {epoch}, step {step}")

                    prompt2     = self.modelmoment2.prompt.prompt
                    if torch.isnan(prompt2).any():
                        raise RuntimeError(f"NaN in prompt2 at epoch {epoch}, step {step}")
                    prompt_new2 = self.AE_cls2(prompt2)
                    AE_loss2    = 0.0005 * (torch.norm(prompt_new2 - prompt2) / prompt2.shape[0])**2
                    if torch.isnan(AE_loss2).any():
                        raise RuntimeError(f"NaN in AE_loss2 at epoch {epoch}, step {step}")

                    # Reconstruction losses 
                    s0, s1, s2 = trg_x.shape
                    randuniform = torch.empty(s0, s1, s2, device=self.device).uniform_(0, 1)
                    mt = torch.bernoulli(randuniform) 
                    total_elements = s0 * s1 * s2
                    c_mask   = mt.sum() / total_elements
                    c_unmask = (total_elements - mt.sum()) / total_elements
                    src2 = trg_x * (1 - mt)

                    out_recon1  = self.modelmoment.forward_reconstruct(src2)
                    rec1        = out_recon1.reconstruction
                    out_recon2  = self.modelmoment2.forward_reconstruct(src2)
                    rec2        = out_recon2.reconstruction

                    gamma = 0.5
                    loss_rec1  = gamma * c_mask * self.mse_loss(rec1, src2) + (1 - gamma) * c_unmask * self.mse_loss(rec1, src2)
                    loss_rec2  = gamma * c_mask * self.mse_loss(rec2, src2) + (1 - gamma) * c_unmask * self.mse_loss(rec2, src2)
                    if torch.isnan(loss_rec1).any():
                        raise RuntimeError(f"NaN in loss_rec1 at epoch {epoch}, step {step}")
                    if torch.isnan(loss_rec2).any():
                        raise RuntimeError(f"NaN in loss_rec2 at epoch {epoch}, step {step}")

                    # Total:
                    total_loss1 = kd_weight * loss_kl1 + AE_loss1 + loss_rec1
                    total_loss2 = kd_weight * loss_kl2 + AE_loss2 + loss_rec2
                    # combined_loss = total_loss1 + total_loss2
                    if torch.isnan(total_loss1).any():
                        raise RuntimeError(f"NaN in total_loss1 at epoch {epoch}, step {step}")
                    if torch.isnan(total_loss2).any():
                        raise RuntimeError(f"NaN in total_loss2 at epoch {epoch}, step {step}")

                
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()
                self.optimizerAE.zero_grad()
                self.optimizerAE2.zero_grad()

                total_loss1.backward()
                total_loss2.backward()

                self.optimizer.step()
                self.optimizer2.step()
                self.optimizerAE.step()
                self.optimizerAE2.step()
                
                losses = {
                    'KL_loss1': loss_kl1.detach().item(),
                    'AE_loss': AE_loss1.detach().item(),
                    'Loss_reconstruct': loss_rec1.detach().item(),
                    'Total_loss': total_loss1.detach().item(),
                    'KL_loss2': loss_kl2.detach().item(),
                    'AE_loss2': AE_loss2.detach().item(),
                    'Loss_reconstruct2': loss_rec2.detach().item(),
                    'Total_loss2': total_loss2.detach().item()
                }
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)
                

            
            self.lr_scheduler.step()
            self.lr_scheduler2.step()
            self.lr_schedulerAE.step()
            self.lr_schedulerAE2.step()
            torch.cuda.empty_cache()
            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug('-------------------------------------')
            self.calc_metrics(trg_dataloader_test, logger)
        

        self.model1 = deepcopy(self.modelmoment.state_dict())
        self.model2 = deepcopy(self.modelmoment2.state_dict())
        return self.model1, self.model2

    

    def loss_coteaching3(self, y_1, y_2, t):
        num_batch = y_1.shape[0]
        loss_1 = F.cross_entropy(y_1, t)
        loss_2 = F.cross_entropy(y_2, t)
        return torch.sum(loss_1) / num_batch, torch.sum(loss_2) / num_batch

    def loss_coteaching4(self, y_1, y_2, t1, t2):
        num_batch = y_1.shape[0]
        loss_1 = F.cross_entropy(y_1, t1)
        loss_2 = F.cross_entropy(y_2, t2)
        return torch.sum(loss_1) / num_batch, torch.sum(loss_2) / num_batch

    def eval_aggr(self, test_loader):

        model = self.modelmoment
        model2 = self.modelmoment2
        model.eval()
        model2.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                outputs = model(data)
                predictions = outputs.logits
                outputs2 = model2(data)
                predictions2 = outputs2.logits

                max_output, _ = torch.max(predictions, 1)
                max_output2, _ = torch.max(predictions2, 1)

                a = max_output/(max_output+max_output2)
                b = max_output2/(max_output+max_output2)

                # Aggregate output 
                output_target = torch.unsqueeze(a,1)*predictions + torch.unsqueeze(b,1)*predictions2

                # compute loss
                loss = F.cross_entropy(output_target, labels)
                total_loss.append(loss.item())
                pred = output_target.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss_eval = torch.tensor(total_loss).mean()  # average loss
        self.full_preds_eval = torch.cat((preds_list))
        self.full_labels_eval = torch.cat((labels_list))
        # print(f'self.full_preds_eval:{self.full_preds_eval}')

    def calc_metrics(self, test_loader, logger):
       
        self.eval_aggr(test_loader)
        
        # accuracy  
        acc = self.acc(self.full_preds_eval.argmax(dim=1), self.full_labels_eval).item()
        # f1
        f1 = self.f1(self.full_preds_eval.argmax(dim=1), self.full_labels_eval).item()
        # auroc 
        auroc = self.auroc(self.full_preds_eval, self.full_labels_eval).item()

        # print(f'acc\t:{acc}')
        logger.debug(f'f1\t:{f1}')
        # print(f'auroc\t:{auroc}')

        return acc, f1, auroc



# HardLabel Loss Target
class B2TSDA_COT_target_(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_COT_target, self).__init__(configs)

        
        self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls = nn.DataParallel(self.AE_cls)
        self.AE_cls = self.AE_cls.module

        self.AE_cls2 = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        self.AE_cls2 = self.AE_cls2.module

        

        self.best_model_net1 = True
        self.pretrained_source = False

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        self.modelmoment2 = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment2.init()

        # print(f'self.sourceModel:{self.sourceModel}')

        # self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module
        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module
        self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        self.modelmoment2 = self.modelmoment2.module


        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.modelmoment2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE = torch.optim.Adam(
            self.AE_cls.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE2 = torch.optim.Adam(
            self.AE_cls2.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )


        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerRefine = StepLR(self.optimizerRefine, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerRefine2 = StepLR(self.optimizerRefine2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        # --- Self-distillation teacher buffers for two target models ---
        # They will be initialized on the first call to update()
        self.ps_buffer1 = None  # for self.modelmoment
        self.ps_buffer2 = None  # for self.modelmoment2

    def update(self, trg_dataloader, avg_meter, logger, source_model_dir):

        # defining best and last model
        best_src_risk = float('inf')
        # self.best_model = self.modelmoment.state_dict()
        # self.last_model = self.modelmoment.state_dict()
        self.model1 = self.modelmoment.state_dict() # COT
        self.model2 = self.modelmoment2.state_dict()
        self.source_model_dir = source_model_dir

        # for k, v in self.classifier.named_parameters():
        # freeze both classifier and ood detector
        # for k, v in self.classifier.named_parameters():
        #     v.requires_grad = False
        # for k, v in self.temporal_verifier.named_parameters():
        #     v.requires_grad = False

        # print(f'self.pretrained_source:{self.pretrained_source}')

        if not self.pretrained_source:
            print(f'Load pretrained source model..')
            load_source_model_path = source_model_dir + "/checkpoint.pt"
            print(f'source model path:{load_source_model_path}')
            self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
            for param in self.sourceModel.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')

        total_epochs = self.hparams["num_epochs"] 

        # Initialize teacher buffers if not already done.
        num_samples = len(trg_dataloader.dataset)
        if self.ps_buffer1 is None:
            self.ps_buffer1 = torch.zeros(num_samples, self.configs.num_classes).to(self.device)
        if self.ps_buffer2 is None:
            self.ps_buffer2 = torch.zeros(num_samples, self.configs.num_classes).to(self.device)

        # gma = 0.7
        gma = 1
        

        # train
        for epoch in range(0, total_epochs):
            
            self.modelmoment.train()
            self.modelmoment2.train()

            for step, (trg_x, _, trg_idx, _, _) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)

                # Validation data
                # val_x, val_y, _ = next(iter(val_dataloader))
                # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)

                

                # # Co-teaching model output
                # with torch.no_grad():
                #     outputs_ind = self.modelmoment2(trg_x)
                #     trg_pred_ind = outputs_ind.logits
                #     trg_ind_prob = nn.Softmax(dim=1)(trg_pred_ind)

                # Target Model1 output
                outputs = self.modelmoment(trg_x)
                trg_pred = outputs.logits

                # Target Model2 output
                outputs2 = self.modelmoment2(trg_x)
                trg_pred2 = outputs2.logits

                # co-teaching loss
                # loss_1, loss_2 = self.loss_coteaching(trg_pred, trg_pred2, pseudo_labels, rate_schedule[epoch], trg_idx)
                # loss_1, loss_2 = self.loss_coteaching2(trg_pred, trg_pred2, pseudo_labels, rate_schedule[epoch], alpha)
                loss_1, loss_2 = self.loss_coteaching3(trg_pred, trg_pred2, pseudo_labels)

                # reconstruct -------------------------------------------------------------
                # mt = random.uniform(0,1) #mask
                s0,s1,s2 = trg_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                mt = torch.bernoulli(randuniform).to(self.device)
                # m_ones = torch.ones(s0,s1,s2).to(self.device)

                # sum_mt = mt.flatten().sum()
                # # c_mask = 1/((s0*s1*s2) * sum_mt)
                # # c_unmask = 1/(((s0*s1*s2) - sum_mt)*(s0*s1*s2))
                # c_mask = sum_mt/(s0*s1*s2)
                # c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)
                # print(f'c_mask:{c_mask}')
                # print(f'c_unmask:{c_unmask}')
                # print(f'mt:{mt}')
                # print(f'trg_x.shape:{trg_x.shape}')

                total_elements = s0 * s1 * s2
                c_mask = mt.sum() / total_elements
                c_unmask = (total_elements - mt.sum()) / total_elements

                # print(f'c_mask:{c_mask}')
                # print(f'c_mask2:{c_mask2}')

                #src if mt=1 -> x=0
                # src2 = torch.clone(trg_x)
                # src2 = src2 * (m_ones-mt)
                # print(f'src2:{src2}')

                src2 = trg_x * (1 - mt)
                # print(f'src3:{src3}')

                gamma = 0.5
                criterion = RMSELoss()
                # print(f'src2.shape:{src2.shape}')

                # pred reconstruct
                out = self.modelmoment.forward_reconstruct(src2)
                pred_reconstruct = out.reconstruction

                out2 = self.modelmoment2.forward_reconstruct(src2)
                pred_reconstruct2 = out2.reconstruction
                # print(f'pred_reconstruct.shape:{pred_reconstruct.shape}')
                # print(f'src2.shape:{src2.shape}')
                # print(f'criterion(pred_reconstruct, src2):{criterion(pred_reconstruct, src2)}')
                #loss reconstruct
                loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)
                loss_reconstruct2 = gamma * c_mask * criterion(pred_reconstruct2, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct2, src2)
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # -------------------------------------------------------------------------

                # prompt reconstruction -------------------------------------------------------------------------
                AE_loss = 0
                AE_loss2 = 0
                # use prompt
                prompt = self.modelmoment.prompt.prompt
                prompt_new = self.AE_cls(prompt)
                prompt2 = self.modelmoment2.prompt.prompt
                prompt_new2 = self.AE_cls2(prompt2)

                AE_loss = 0.0005 * torch.pow(torch.linalg.norm(prompt_new - prompt)/prompt.shape[0], 2)
                AE_loss2 = 0.0005 * torch.pow(torch.linalg.norm(prompt_new2 - prompt2)/prompt2.shape[0], 2)
                # print(f'AE_loss:{AE_loss}')

                # while AE_loss > 5:
                #     AE_loss = AE_loss / 10

                # while AE_loss2 > 5:
                #     AE_loss2 = AE_loss2 / 10
                
                # -----------------------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                loss = loss_1 + AE_loss + loss_reconstruct
                loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                # loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.optimizer2.zero_grad()
                loss2.backward()
                self.optimizer2.step()

                self.optimizerAE.step()
                self.optimizerAE2.step()
                # self.decoder_optimizer.step()
                # self.global_step+=1


                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                 'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)


            # Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            

            self.lr_scheduler.step()
            self.lr_scheduler2.step()
            self.lr_schedulerAE.step()
            self.lr_schedulerAE2.step()

            # # saving the best model based on src risk
            # if (epoch + 1) % 10 == 0 and (loss_1.avg < best_src_risk or loss_2.avg < best_src_risk):
            #     # best_src_risk = avg_meter['Total_loss'].avg

            #     if loss_1.avg < loss_2.avg:
            #         best_src_risk = loss_1.avg
            #         self.best_model = deepcopy(self.modelmoment.state_dict())
            #         self.best_model_net1 = True
            #     else:
            #         best_src_risk = loss_2.avg
            #         self.best_model = deepcopy(self.modelmoment2.state_dict())
            #         self.best_model_net1 = False
            

            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        self.model1 = deepcopy(self.modelmoment.state_dict())
        self.model2 = deepcopy(self.modelmoment2.state_dict())
        # return self.last_model, self.best_model
        return self.model1, self.model2

    def loss_coteaching(self, y_1, y_2, t, forget_rate, ind):
        loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
        ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
        ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))
        # print(f'num_remember:{num_remember}')

        # print(f'noise_or_not.type():{type(noise_or_not)}')
        # print(f'num_remember.type():{type(num_remember)}')
        # print(f'ind type:{type(ind)}')
        # print(f'ind_1_sorted type:{type(ind_1_sorted)}')

        # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
        # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

    def loss_coteaching2(self, y_1, y_2, t, forget_rate, alpha):
        loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
        ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
        ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]
        # print(f'loss_2_sorted.shape:{loss_2_sorted.shape}')
        # print(f'y_1.shape:{y_1.shape}')
        # print(f'y_1.shape[0]:{y_1.shape[0]}')

        remember_rate = 1 - (1-alpha) * forget_rate
        num_remember = int(remember_rate * len(loss_2_sorted))
        # print(f'num_remember:{num_remember}')

        # print(f'noise_or_not.type():{type(noise_or_not)}')
        # print(f'num_remember.type():{type(num_remember)}')
        # print(f'ind type:{type(ind)}')
        # print(f'ind_1_sorted type:{type(ind_1_sorted)}')

        # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
        # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

    def loss_coteaching3(self, y_1, y_2, t):
        # loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
        # ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
        # loss_1_sorted = loss_1[ind_1_sorted]

        # loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
        # ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
        # loss_2_sorted = loss_2[ind_2_sorted]

        # remember_rate = 1 - (1-alpha) * forget_rate
        num_batch = y_1.shape[0]
        # print(f'num_remember:{num_remember}')

        # print(f'noise_or_not.type():{type(noise_or_not)}')
        # print(f'num_remember.type():{type(num_remember)}')
        # print(f'ind type:{type(ind)}')
        # print(f'ind_1_sorted type:{type(ind_1_sorted)}')

        # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
        # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

        # ind_1_update=ind_1_sorted[:num_remember]
        # ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1 = F.cross_entropy(y_1, t)
        loss_2 = F.cross_entropy(y_2, t)

        return torch.sum(loss_1)/num_batch, torch.sum(loss_2)/num_batch

    def refine(self, trg_dataloader, avg_meter, logger, modelmoment, optimizer, lr_scheduler):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = modelmoment.state_dict()
        self.last_model = modelmoment.state_dict()
        # self.source_model_dir = source_model_dir

        total_epochs = self.hparams["num_iter"] 
        # w = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)
        # print(f'w.shape:{w.shape}')

        # train
        for epoch in range(0, total_epochs):
        # for epoch in range(1, self.hparams["num_epochs"] + 1):

            # non_candidate_loss = 0
            modelmoment.train()

            for step, (trg_x, _, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)

                # Validation data
                # val_x, val_y, _ = next(iter(val_dataloader))
                # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)


                # Target Model output
                outputs = modelmoment(trg_x)
                trg_pred = outputs.logits

                trg_prob = nn.Softmax(dim=1)(trg_pred)
                # trg_conf, _ = torch.max(trg_prob, 1)
                # print(f'trg_prob:{trg_prob}')


                # Target Model output with strong aug
                outputs_strong = modelmoment(trg_x_strong)
                trg_pred_strong = outputs_strong.logits

                trg_prob_strong = nn.Softmax(dim=1)(trg_pred_strong)
                trg_conf_strong, _ = torch.max(trg_prob_strong, 1)

                # Target Model output with weak aug
                outputs_weak = modelmoment(trg_x_weak)
                trg_pred_weak = outputs_weak.logits

                trg_prob_weak = nn.Softmax(dim=1)(trg_pred_weak)
                trg_conf_weak, _ = torch.max(trg_prob_weak, 1)
                # print(f'trg_prob_weak:{trg_prob_weak}')

                # print(f'trg_x_strong.shape:{trg_x_strong.shape}')
                # print(f'trg_x_strong:{trg_x_strong}')
                # print(f'trg_pred_strong:{trg_pred_strong}')
                # print(f'trg_prob_strong:{trg_prob_strong}')


                # select pseudo-candidate set (Z) based on threshold -----------------------------------------------
                prob_sel = trg_prob > self.hparams["tau"] 
                non_prob_sel = ~prob_sel              
                # calculate norm |Z|
                num_candidate = torch.sum(prob_sel, dim=1)
                # print(f'num_candidate:{num_candidate}')

                # select samples based on threshold
                # conf_sel = trg_conf > self.hparams["tau"]
                # non_conf_sel = ~conf_sel


                # print(f'prob_sel:{prob_sel}')
                # print(f'non_prob_sel:{non_prob_sel}')
                # print(f'conf_sel:{conf_sel}')
                # print(f'len(conf_sel):{len(conf_sel)}')
                # print(f'range(len(conf_sel)):{range(len(conf_sel))}')
                # print(f'~conf_sel:{~conf_sel}')
                # print(f'trg_conf.shape:{trg_conf.shape}')

                ind_candidate = torch.argwhere(prob_sel)
                ind_non_candidate = torch.argwhere(non_prob_sel)

                # ind_cand_sampl = torch.argwhere(conf_sel)
                # ind_non_cand_sampl = torch.argwhere(non_conf_sel)
                
                # print(f'ind_candidate:{ind_candidate}')
                # print(f'ind_cand_sampl:{ind_cand_sampl}')
                # print(f'ind_non_candidate:{ind_non_candidate}')

                # create w
                # w = torch.zeros(trg_prob.shape).to(self.device)

                # print(f'w:{w}')
                probs_strong = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'trg_prob_strong[0]:{trg_prob_strong[0]}')

                # initialized w
                if epoch == 0:
                    w = torch.zeros(trg_prob.shape).to(self.device)
                    for i in ind_candidate:
                        w[i[0],i[1]] = 1/num_candidate[i[0]]
                        # print(f'i:{i}')
                        # print(f'num_candidate[i]:{num_candidate[i]}')
                else:
                    for i in ind_candidate:
                        probs_strong[i[0],i[1]] = trg_prob_strong[i[0],i[1]]

                    for i in ind_candidate:
                        w[i[0],i[1]] = trg_prob_strong[i[0],i[1]]/torch.sum(trg_prob_strong[i[0]])

                # candidate_sel = trg_conf_strong[ind_cand_sampl] < self.hparams["threshold"]
                # print(f'after intialized w:{w}')

                candidate_sel = trg_conf_strong < self.hparams["threshold"]

                ind_candidate_sel = torch.logical_and(conf_sel, candidate_sel)
                ind_candidate_loss = torch.squeeze(ind_candidate_sel.nonzero(), dim=-1)
                # print(f'ind_candidate_loss2:{ind_candidate_loss2}')
                # print(f'ind_keep:{ind_keep}')

                # print(f'trg_conf_strong:{trg_conf_strong}')
                # print(f'trg_conf:{trg_conf}')
                # print(f'trg_conf_strong[ind_cand_sampl]:{trg_conf_strong[ind_cand_sampl]}')
                # print(f'trg_conf_strong[ind_candidate]:{trg_conf_strong[ind_candidate]}')
                # candidate_sel = trg_conf_strong[ind_candidate] < self.hparams["threshold"]
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'trg_prob_strong[ind_candidate]:{trg_prob_strong[ind_candidate]}')
                # print(f'trg_conf_strong:{trg_conf_strong}')
                # print(f'candidate_sel:{candidate_sel}')
                # if epoch > 0:
                # print(f'trg_x_strong:{trg_x_strong}')
                # print(f'ind_candidate:{ind_candidate}')
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'probs_strong:{probs_strong}')
                # print(f'w:{w}')

                # ind_candidate_loss = torch.argwhere(candidate_sel)
                # ind_candidate_loss = torch.argwhere(torch.squeeze(candidate_sel, dim=-1))
                # print(f'ind_candidate_loss:{ind_candidate_loss}')
                # print(f'w[ind_candidate_loss]:{w[ind_candidate_loss]}')
                # print(f'trg_prob_strong[ind_candidate_loss]:{trg_prob_strong[ind_candidate_loss]}')

                # print(f'w[ind_candidate_loss]:{w[ind_candidate_loss]}')
                # print(f'trg_prob_strong[ind_candidate_loss].shape:{trg_prob_strong[ind_candidate_loss].shape}')

                # log_w = np.log(w.cpu())
                # print(f'w[torch.squeeze(ind_candidate_loss, dim=-1)]:{w[torch.squeeze(ind_candidate_loss, dim=-1)]}')
                # print(f'w:{w}')
                # print(f'w[ind_candidate_loss].shape:{w[ind_candidate_loss].shape}')
                # print(f'F.log_softmax(w[ind_candidate_loss]):{F.log_softmax(w[ind_candidate_loss], dim=1)}')
                # print(f'nn.Softmax(w[ind_candidate_loss]):{nn.Softmax(dim=-1)(w[ind_candidate_loss])}')
                if ind_candidate_loss.numel():
                    # print(f'ind_candidate_loss:{ind_candidate_loss}')
                    # print(f'w[ind_candidate_loss].shape:{w[ind_candidate_loss].shape}')
                    # print(f'trg_prob_strong[ind_candidate_loss].shape:{trg_prob_strong[ind_candidate_loss].shape}')
                    # print(f'w[torch.squeeze(ind_candidate_loss, dim=-1)].shape:{w[torch.squeeze(ind_candidate_loss, dim=-1)].shape}')
                    # print(f'trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)].shape:{trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)].shape}')
                    # candidate_loss = self.kl_loss(F.log_softmax(w[torch.squeeze(ind_candidate_loss, dim=-1)], dim=1), trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)])
                    candidate_loss = self.kl_loss(F.log_softmax(w[ind_candidate_loss], dim=1), trg_prob_strong[ind_candidate_loss])
                else:
                    candidate_loss = torch.zeros(1).to(self.device)
                    # candidate_loss = 0.0
                # candidate_loss = self.kl_loss(F.log_softmax(w, dim=1), trg_prob_strong)
                # candidate_loss = self.kl_loss(trg_prob_strong[ind_candidate_loss], F.log_softmax(w[ind_candidate_loss], -1))
                # candidate_loss = self.kl_loss(nn.Softmax(dim=-1)(w[ind_candidate_loss]), trg_prob_strong[ind_candidate_loss])
                # candidate_loss = self.kl_loss(trg_prob_strong[ind_candidate_loss], w[ind_candidate_loss])
                # print(f'candidate_loss:{candidate_loss}')
                # print(f'ind_non_candidate:{ind_non_candidate}')

                # print(f'trg_prob_weak[ind_non_candidate]:{trg_prob_weak[ind_non_candidate]}')

                # for i in ind_non_candidate:
                #     # print(f'i[0]:{i[0]}')
                #     # print(f'i[1]:{i[1]}')
                #     print(f'trg_prob_weak[i[0],i[1]]:{trg_prob_weak[i[0],i[1]]}')
                # print(f'trg_prob_weak:{trg_prob_weak}')
                # print(f'trg_pred_weak:{trg_pred_weak}')
                # non_candidate_loss = 0
                non_candidate_loss = torch.zeros(1).to(self.device)
                for i in ind_non_candidate:
                    non_candidate_loss += torch.log(1 - trg_prob_weak[i[0],i[1]]).to(self.device)
                    # non_candidate_loss += torch.log(1 - trg_pred_weak[i[0],i[1]])
                    # print(f'trg_pred_weak[{i[0]},{i[1]}]:{trg_pred_weak[i[0],i[1]]}')
                    # print(f'torch.log(1 - trg_prob_weak[{i[0]},{i[1]}]):{torch.log(1 - trg_prob_weak[i[0],i[1]])}')
                    # print(f'torch.log(1 - trg_pred_weak[{i[0]},{i[1]}]):{torch.log(1 - trg_pred_weak[i[0],i[1]])}')
                    # print(f'non_candidate_loss:{non_candidate_loss}')

                non_candidate_loss = -non_candidate_loss/self.hparams["batch_size"]
                # print(f'average -non_candidate_loss:{non_candidate_loss}')
                # w = torch.div(torch.t(conf_sel),num_candidate)
                # w = torch.t(w)
                # w[ind_candidate] = 1/num_candidate
                # print(f'after initialize w:{w}')
                # print(f'len(trg_x):{len(trg_x)}')
                # pseudo_candidate_set = trg_x[ind_candidate]
                # pseudo_labels = pseudo_labels[ind_candidate]
                # print(f'after ind_keep len(trg_x):{len(trg_x)}')
                # ind_keep = torch.squeeze(conf_sel.nonzero(), dim=-1)
                # print(f'conf_sel:{conf_sel}')
                # print(f'ind_keep:{ind_keep}')
                # end select pseudo-labels based on threshold ------------------------------------------
                

                # # CE_loss = self.cross_entropy(outputs_sorted.logits, pseudo_labels_sorted)
                # # CE_loss = self.cross_entropy(output_target[ind_loss_update], pseudo_labels_sorted)
                # CE_loss = self.cross_entropy(trg_pred, pseudo_labels)
                # # print(f'CE_loss:{CE_loss}')
                # # end Lama -----------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                # loss_refine = candidate_loss + self.hparams["lam"] * (non_candidate_loss)
                loss_refine = candidate_loss + self.hparams["lam"] * (non_candidate_loss)
                # loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                # loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')
                

                # self.optimizer.zero_grad()
                # loss_refine.backward()
                # self.optimizer.step()
                # self.optimizerRefine.zero_grad()
                optimizer.zero_grad()
                loss_refine.backward()
                optimizer.step()
                # self.optimizerRefine.step()

                
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                losses_refine = {'candidate_loss': candidate_loss.detach().item(), 'non_candidate_loss': non_candidate_loss.detach().item(), 'Total_loss': loss_refine.detach().item()}
                # losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                #  'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses_refine.items():
                    avg_meter[key].update(val, 32)


            # Update w



            # Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # self.lr_scheduler.step()
            # self.lr_schedulerRefine.step()
            lr_scheduler.step()



            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_iter"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def refine2(self, trg_dataloader, avg_meter, logger, modelmoment, optimizer, lr_scheduler):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = modelmoment.state_dict()
        self.last_model = modelmoment.state_dict()

        total_epochs = self.hparams["num_iter"]

        for epoch in range(total_epochs):
            modelmoment.train()

            for step, (trg_x, trg_y, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                trg_y = trg_y.long().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)

                # Source Model predictions for target data
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits
                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)

                # Target model outputs (original, strong, weak augmentations)
                outputs = modelmoment(trg_x)
                trg_pred = outputs.logits
                trg_prob = nn.Softmax(dim=1)(trg_pred)
                trg_conf, _ = torch.max(trg_prob, 1)

                outputs_strong = modelmoment(trg_x_strong)
                trg_pred_strong = outputs_strong.logits
                trg_prob_strong = nn.Softmax(dim=1)(trg_pred_strong)
                trg_conf_strong, _ = torch.max(trg_prob_strong, 1)

                outputs_weak = modelmoment(trg_x_weak)
                trg_pred_weak = outputs_weak.logits
                trg_prob_weak = nn.Softmax(dim=1)(trg_pred_weak)
                trg_conf_weak, _ = torch.max(trg_prob_weak, 1)

                # Candidate set selection based on threshold tau
                # prob_sel: Boolean mask of shape (batch_size, num_classes)
                prob_sel = trg_prob > self.hparams["tau"]
                # num_candidate: number of candidate classes per sample (shape: [batch_size])
                num_candidate = torch.sum(prob_sel, dim=1)

                batch_size, num_classes = trg_prob.shape

                # Vectorized initialization or update of the label confidence vector w
                if epoch == 0:
                    # Initialize w: assign uniform probability over candidate classes
                    w = torch.zeros_like(trg_prob)
                    # For candidate positions, assign 1/num_candidate per sample using broadcasting
                    w[prob_sel] = 1.0 / num_candidate.view(-1, 1).expand_as(trg_prob)[prob_sel]
                else:
                    # Update w using strong predictions for candidate classes only.
                    # Compute, per sample, the sum of strong predictions for candidate classes:
                    candidate_sum = torch.sum(trg_prob_strong * prob_sel.float(), dim=1)  # shape: [batch_size]
                    # Create a new w vector that will update candidate positions only where candidate_sum > 0
                    w_new = w.clone()
                    update_mask = candidate_sum > 0  # boolean mask for samples to update
                    if update_mask.any():
                        # For samples where update_mask is True, compute new candidate values:
                        # Divide strong predictions by candidate_sum (broadcasted) and zero out non-candidate positions.
                        w_update = (trg_prob_strong[update_mask] / candidate_sum[update_mask].view(-1, 1)) * prob_sel[update_mask].float()
                        w_new[update_mask] = w_update
                    # For samples where candidate_sum == 0, retain the previous w values.
                    w = w_new.detach()

                print(f'w:{w}')
                print(f'trg_y[prob_sel]:{trg_y[prob_sel]}')

                # Candidate loss: select positions where strong confidence is below a threshold
                candidate_mask = (trg_conf_strong < self.hparams["threshold"]) & (trg_conf > self.hparams["tau"])
                # print(f'candidate_mask:{candidate_mask}')
                if candidate_mask.sum() > 0:
                    candidate_loss = self.kl_loss(F.log_softmax(w[candidate_mask], dim=1),
                                                  trg_prob_strong[candidate_mask])
                else:
                    candidate_loss = torch.zeros(1).to(self.device)

                # Non-candidate loss: vectorized computation over positions not in the candidate set
                non_prob_mask = ~prob_sel  # positions outside the pseudo-candidate set
                if non_prob_mask.sum() > 0:
                    non_candidate_loss = - torch.log(1 - trg_prob_weak[non_prob_mask]).sum() / batch_size
                else:
                    non_candidate_loss = torch.zeros(1).to(self.device)

                # Overall loss
                loss_refine = candidate_loss + self.hparams["lam"] * non_candidate_loss

                optimizer.zero_grad()
                loss_refine.backward()
                optimizer.step()

                losses_refine = {'candidate_loss': candidate_loss.detach().item(),
                                 'non_candidate_loss': non_candidate_loss.detach().item(),
                                 'Total_loss': loss_refine.detach().item()}
                for key, val in losses_refine.items():
                    avg_meter[key].update(val, batch_size)

            lr_scheduler.step()

            logger.debug(f'[Epoch : {epoch+1}/{total_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug('-------------------------------------')

        return self.last_model, self.best_model

class B2TSDA_COT_target__(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_COT_target, self).__init__(configs)

        
        self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls = nn.DataParallel(self.AE_cls)
        self.AE_cls = self.AE_cls.module

        self.AE_cls2 = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        self.AE_cls2 = self.AE_cls2.module

        

        self.best_model_net1 = True
        self.pretrained_source = False

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        self.modelmoment2 = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment2.init()

        # print(f'self.sourceModel:{self.sourceModel}')

        # self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module
        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module
        self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        self.modelmoment2 = self.modelmoment2.module


        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer2 = torch.optim.Adam(
            self.modelmoment2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE = torch.optim.Adam(
            self.AE_cls.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerAE2 = torch.optim.Adam(
            self.AE_cls2.parameters(),
            lr=hparams["learning_rate_AE"],
            weight_decay=hparams["weight_decay"]
        )


        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerRefine = StepLR(self.optimizerRefine, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerRefine2 = StepLR(self.optimizerRefine2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def update(self, trg_dataloader, avg_meter, logger, source_model_dir):

        # defining best and last model
        best_src_risk = float('inf')
        # self.best_model = self.modelmoment.state_dict()
        # self.last_model = self.modelmoment.state_dict()
        self.model1 = self.modelmoment.state_dict() # COT
        self.model2 = self.modelmoment2.state_dict()
        self.source_model_dir = source_model_dir

        # for k, v in self.classifier.named_parameters():
        # freeze both classifier and ood detector
        # for k, v in self.classifier.named_parameters():
        #     v.requires_grad = False
        # for k, v in self.temporal_verifier.named_parameters():
        #     v.requires_grad = False

        # print(f'self.pretrained_source:{self.pretrained_source}')

        if not self.pretrained_source:
            print(f'Load pretrained source model..')
            load_source_model_path = source_model_dir + "/checkpoint.pt"
            print(f'source model path:{load_source_model_path}')
            self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
            for param in self.sourceModel.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')

        total_epochs = self.hparams["num_epochs"] 
        

        # train
        for epoch in range(0, total_epochs):
            
            self.modelmoment.train()
            self.modelmoment2.train()

            for step, (trg_x, _, trg_idx, _, _) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)

                # Validation data
                # val_x, val_y, _ = next(iter(val_dataloader))
                # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)

                

                # Co-teaching model output
                with torch.no_grad():
                    outputs_ind = self.modelmoment2(trg_x)
                    trg_pred_ind = outputs_ind.logits
                    trg_ind_prob = nn.Softmax(dim=1)(trg_pred_ind)

                # Target Model1 output
                outputs = self.modelmoment(trg_x)
                trg_pred = outputs.logits

                # Target Model2 output
                outputs2 = self.modelmoment2(trg_x)
                trg_pred2 = outputs2.logits

                # co-teaching loss
                # loss_1, loss_2 = self.loss_coteaching(trg_pred, trg_pred2, pseudo_labels, rate_schedule[epoch], trg_idx)
                # loss_1, loss_2 = self.loss_coteaching2(trg_pred, trg_pred2, pseudo_labels, rate_schedule[epoch], alpha)
                loss_1, loss_2 = self.loss_coteaching3(trg_pred, trg_pred2, pseudo_labels)

                # reconstruct -------------------------------------------------------------
                # mt = random.uniform(0,1) #mask
                s0,s1,s2 = trg_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                mt = torch.bernoulli(randuniform).to(self.device)
                # m_ones = torch.ones(s0,s1,s2).to(self.device)

                # sum_mt = mt.flatten().sum()
                # # c_mask = 1/((s0*s1*s2) * sum_mt)
                # # c_unmask = 1/(((s0*s1*s2) - sum_mt)*(s0*s1*s2))
                # c_mask = sum_mt/(s0*s1*s2)
                # c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)
                # print(f'c_mask:{c_mask}')
                # print(f'c_unmask:{c_unmask}')
                # print(f'mt:{mt}')
                # print(f'trg_x.shape:{trg_x.shape}')

                total_elements = s0 * s1 * s2
                c_mask = mt.sum() / total_elements
                c_unmask = (total_elements - mt.sum()) / total_elements

                # print(f'c_mask:{c_mask}')
                # print(f'c_mask2:{c_mask2}')

                #src if mt=1 -> x=0
                # src2 = torch.clone(trg_x)
                # src2 = src2 * (m_ones-mt)
                # print(f'src2:{src2}')

                src2 = trg_x * (1 - mt)
                # print(f'src3:{src3}')

                gamma = 0.5
                criterion = RMSELoss()
                # print(f'src2.shape:{src2.shape}')

                # pred reconstruct
                out = self.modelmoment.forward_reconstruct(src2)
                pred_reconstruct = out.reconstruction

                out2 = self.modelmoment2.forward_reconstruct(src2)
                pred_reconstruct2 = out2.reconstruction
                # print(f'pred_reconstruct.shape:{pred_reconstruct.shape}')
                # print(f'src2.shape:{src2.shape}')
                # print(f'criterion(pred_reconstruct, src2):{criterion(pred_reconstruct, src2)}')
                #loss reconstruct
                loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)
                loss_reconstruct2 = gamma * c_mask * criterion(pred_reconstruct2, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct2, src2)
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # -------------------------------------------------------------------------

                # prompt reconstruction -------------------------------------------------------------------------
                AE_loss = 0
                AE_loss2 = 0
                # use prompt
                prompt = self.modelmoment.prompt.prompt
                prompt_new = self.AE_cls(prompt)
                prompt2 = self.modelmoment2.prompt.prompt
                prompt_new2 = self.AE_cls2(prompt2)

                AE_loss = 0.0005 * torch.pow(torch.linalg.norm(prompt_new - prompt)/prompt.shape[0], 2)
                AE_loss2 = 0.0005 * torch.pow(torch.linalg.norm(prompt_new2 - prompt2)/prompt2.shape[0], 2)
                # print(f'AE_loss:{AE_loss}')

                # while AE_loss > 5:
                #     AE_loss = AE_loss / 10

                # while AE_loss2 > 5:
                #     AE_loss2 = AE_loss2 / 10
                
                # -----------------------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                loss = loss_1 + AE_loss + loss_reconstruct
                loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                # loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.optimizer2.zero_grad()
                loss2.backward()
                self.optimizer2.step()

                self.optimizerAE.step()
                self.optimizerAE2.step()
                # self.decoder_optimizer.step()
                # self.global_step+=1


                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                 'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)


            # Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            

            self.lr_scheduler.step()
            self.lr_scheduler2.step()
            self.lr_schedulerAE.step()
            self.lr_schedulerAE2.step()

            # # saving the best model based on src risk
            # if (epoch + 1) % 10 == 0 and (loss_1.avg < best_src_risk or loss_2.avg < best_src_risk):
            #     # best_src_risk = avg_meter['Total_loss'].avg

            #     if loss_1.avg < loss_2.avg:
            #         best_src_risk = loss_1.avg
            #         self.best_model = deepcopy(self.modelmoment.state_dict())
            #         self.best_model_net1 = True
            #     else:
            #         best_src_risk = loss_2.avg
            #         self.best_model = deepcopy(self.modelmoment2.state_dict())
            #         self.best_model_net1 = False
            

            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        self.model1 = deepcopy(self.modelmoment.state_dict())
        self.model2 = deepcopy(self.modelmoment2.state_dict())
        # return self.last_model, self.best_model
        return self.model1, self.model2

    def loss_coteaching(self, y_1, y_2, t, forget_rate, ind):
        loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
        ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
        ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))
        # print(f'num_remember:{num_remember}')

        # print(f'noise_or_not.type():{type(noise_or_not)}')
        # print(f'num_remember.type():{type(num_remember)}')
        # print(f'ind type:{type(ind)}')
        # print(f'ind_1_sorted type:{type(ind_1_sorted)}')

        # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
        # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

    def loss_coteaching2(self, y_1, y_2, t, forget_rate, alpha):
        loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
        ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
        ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]
        # print(f'loss_2_sorted.shape:{loss_2_sorted.shape}')
        # print(f'y_1.shape:{y_1.shape}')
        # print(f'y_1.shape[0]:{y_1.shape[0]}')

        remember_rate = 1 - (1-alpha) * forget_rate
        num_remember = int(remember_rate * len(loss_2_sorted))
        # print(f'num_remember:{num_remember}')

        # print(f'noise_or_not.type():{type(noise_or_not)}')
        # print(f'num_remember.type():{type(num_remember)}')
        # print(f'ind type:{type(ind)}')
        # print(f'ind_1_sorted type:{type(ind_1_sorted)}')

        # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
        # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

    def loss_coteaching3(self, y_1, y_2, t):
        # loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
        # ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()
        # loss_1_sorted = loss_1[ind_1_sorted]

        # loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
        # ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
        # loss_2_sorted = loss_2[ind_2_sorted]

        # remember_rate = 1 - (1-alpha) * forget_rate
        num_batch = y_1.shape[0]
        # print(f'num_remember:{num_remember}')

        # print(f'noise_or_not.type():{type(noise_or_not)}')
        # print(f'num_remember.type():{type(num_remember)}')
        # print(f'ind type:{type(ind)}')
        # print(f'ind_1_sorted type:{type(ind_1_sorted)}')

        # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
        # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

        # ind_1_update=ind_1_sorted[:num_remember]
        # ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1 = F.cross_entropy(y_1, t)
        loss_2 = F.cross_entropy(y_2, t)

        return torch.sum(loss_1)/num_batch, torch.sum(loss_2)/num_batch

    def refine(self, trg_dataloader, avg_meter, logger, modelmoment, optimizer, lr_scheduler):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = modelmoment.state_dict()
        self.last_model = modelmoment.state_dict()
        # self.source_model_dir = source_model_dir

        total_epochs = self.hparams["num_iter"] 
        # w = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)
        # print(f'w.shape:{w.shape}')

        # train
        for epoch in range(0, total_epochs):
        # for epoch in range(1, self.hparams["num_epochs"] + 1):

            # non_candidate_loss = 0
            modelmoment.train()

            for step, (trg_x, _, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)

                # Validation data
                # val_x, val_y, _ = next(iter(val_dataloader))
                # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)


                # Target Model output
                outputs = modelmoment(trg_x)
                trg_pred = outputs.logits

                trg_prob = nn.Softmax(dim=1)(trg_pred)
                # trg_conf, _ = torch.max(trg_prob, 1)
                # print(f'trg_prob:{trg_prob}')


                # Target Model output with strong aug
                outputs_strong = modelmoment(trg_x_strong)
                trg_pred_strong = outputs_strong.logits

                trg_prob_strong = nn.Softmax(dim=1)(trg_pred_strong)
                trg_conf_strong, _ = torch.max(trg_prob_strong, 1)

                # Target Model output with weak aug
                outputs_weak = modelmoment(trg_x_weak)
                trg_pred_weak = outputs_weak.logits

                trg_prob_weak = nn.Softmax(dim=1)(trg_pred_weak)
                trg_conf_weak, _ = torch.max(trg_prob_weak, 1)
                # print(f'trg_prob_weak:{trg_prob_weak}')

                # print(f'trg_x_strong.shape:{trg_x_strong.shape}')
                # print(f'trg_x_strong:{trg_x_strong}')
                # print(f'trg_pred_strong:{trg_pred_strong}')
                # print(f'trg_prob_strong:{trg_prob_strong}')


                # select pseudo-candidate set (Z) based on threshold -----------------------------------------------
                prob_sel = trg_prob > self.hparams["tau"] 
                non_prob_sel = ~prob_sel              
                # calculate norm |Z|
                num_candidate = torch.sum(prob_sel, dim=1)
                # print(f'num_candidate:{num_candidate}')

                # select samples based on threshold
                # conf_sel = trg_conf > self.hparams["tau"]
                # non_conf_sel = ~conf_sel


                # print(f'prob_sel:{prob_sel}')
                # print(f'non_prob_sel:{non_prob_sel}')
                # print(f'conf_sel:{conf_sel}')
                # print(f'len(conf_sel):{len(conf_sel)}')
                # print(f'range(len(conf_sel)):{range(len(conf_sel))}')
                # print(f'~conf_sel:{~conf_sel}')
                # print(f'trg_conf.shape:{trg_conf.shape}')

                ind_candidate = torch.argwhere(prob_sel)
                ind_non_candidate = torch.argwhere(non_prob_sel)

                # ind_cand_sampl = torch.argwhere(conf_sel)
                # ind_non_cand_sampl = torch.argwhere(non_conf_sel)
                
                # print(f'ind_candidate:{ind_candidate}')
                # print(f'ind_cand_sampl:{ind_cand_sampl}')
                # print(f'ind_non_candidate:{ind_non_candidate}')

                # create w
                # w = torch.zeros(trg_prob.shape).to(self.device)

                # print(f'w:{w}')
                probs_strong = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'trg_prob_strong[0]:{trg_prob_strong[0]}')

                # initialized w
                if epoch == 0:
                    w = torch.zeros(trg_prob.shape).to(self.device)
                    for i in ind_candidate:
                        w[i[0],i[1]] = 1/num_candidate[i[0]]
                        # print(f'i:{i}')
                        # print(f'num_candidate[i]:{num_candidate[i]}')
                else:
                    for i in ind_candidate:
                        probs_strong[i[0],i[1]] = trg_prob_strong[i[0],i[1]]

                    for i in ind_candidate:
                        w[i[0],i[1]] = trg_prob_strong[i[0],i[1]]/torch.sum(trg_prob_strong[i[0]])

                # candidate_sel = trg_conf_strong[ind_cand_sampl] < self.hparams["threshold"]
                # print(f'after intialized w:{w}')

                candidate_sel = trg_conf_strong < self.hparams["threshold"]

                ind_candidate_sel = torch.logical_and(conf_sel, candidate_sel)
                ind_candidate_loss = torch.squeeze(ind_candidate_sel.nonzero(), dim=-1)
                # print(f'ind_candidate_loss2:{ind_candidate_loss2}')
                # print(f'ind_keep:{ind_keep}')

                # print(f'trg_conf_strong:{trg_conf_strong}')
                # print(f'trg_conf:{trg_conf}')
                # print(f'trg_conf_strong[ind_cand_sampl]:{trg_conf_strong[ind_cand_sampl]}')
                # print(f'trg_conf_strong[ind_candidate]:{trg_conf_strong[ind_candidate]}')
                # candidate_sel = trg_conf_strong[ind_candidate] < self.hparams["threshold"]
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'trg_prob_strong[ind_candidate]:{trg_prob_strong[ind_candidate]}')
                # print(f'trg_conf_strong:{trg_conf_strong}')
                # print(f'candidate_sel:{candidate_sel}')
                # if epoch > 0:
                # print(f'trg_x_strong:{trg_x_strong}')
                # print(f'ind_candidate:{ind_candidate}')
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'probs_strong:{probs_strong}')
                # print(f'w:{w}')

                # ind_candidate_loss = torch.argwhere(candidate_sel)
                # ind_candidate_loss = torch.argwhere(torch.squeeze(candidate_sel, dim=-1))
                # print(f'ind_candidate_loss:{ind_candidate_loss}')
                # print(f'w[ind_candidate_loss]:{w[ind_candidate_loss]}')
                # print(f'trg_prob_strong[ind_candidate_loss]:{trg_prob_strong[ind_candidate_loss]}')

                # print(f'w[ind_candidate_loss]:{w[ind_candidate_loss]}')
                # print(f'trg_prob_strong[ind_candidate_loss].shape:{trg_prob_strong[ind_candidate_loss].shape}')

                # log_w = np.log(w.cpu())
                # print(f'w[torch.squeeze(ind_candidate_loss, dim=-1)]:{w[torch.squeeze(ind_candidate_loss, dim=-1)]}')
                # print(f'w:{w}')
                # print(f'w[ind_candidate_loss].shape:{w[ind_candidate_loss].shape}')
                # print(f'F.log_softmax(w[ind_candidate_loss]):{F.log_softmax(w[ind_candidate_loss], dim=1)}')
                # print(f'nn.Softmax(w[ind_candidate_loss]):{nn.Softmax(dim=-1)(w[ind_candidate_loss])}')
                if ind_candidate_loss.numel():
                    # print(f'ind_candidate_loss:{ind_candidate_loss}')
                    # print(f'w[ind_candidate_loss].shape:{w[ind_candidate_loss].shape}')
                    # print(f'trg_prob_strong[ind_candidate_loss].shape:{trg_prob_strong[ind_candidate_loss].shape}')
                    # print(f'w[torch.squeeze(ind_candidate_loss, dim=-1)].shape:{w[torch.squeeze(ind_candidate_loss, dim=-1)].shape}')
                    # print(f'trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)].shape:{trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)].shape}')
                    # candidate_loss = self.kl_loss(F.log_softmax(w[torch.squeeze(ind_candidate_loss, dim=-1)], dim=1), trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)])
                    candidate_loss = self.kl_loss(F.log_softmax(w[ind_candidate_loss], dim=1), trg_prob_strong[ind_candidate_loss])
                else:
                    candidate_loss = torch.zeros(1).to(self.device)
                    # candidate_loss = 0.0
                # candidate_loss = self.kl_loss(F.log_softmax(w, dim=1), trg_prob_strong)
                # candidate_loss = self.kl_loss(trg_prob_strong[ind_candidate_loss], F.log_softmax(w[ind_candidate_loss], -1))
                # candidate_loss = self.kl_loss(nn.Softmax(dim=-1)(w[ind_candidate_loss]), trg_prob_strong[ind_candidate_loss])
                # candidate_loss = self.kl_loss(trg_prob_strong[ind_candidate_loss], w[ind_candidate_loss])
                # print(f'candidate_loss:{candidate_loss}')
                # print(f'ind_non_candidate:{ind_non_candidate}')

                # print(f'trg_prob_weak[ind_non_candidate]:{trg_prob_weak[ind_non_candidate]}')

                # for i in ind_non_candidate:
                #     # print(f'i[0]:{i[0]}')
                #     # print(f'i[1]:{i[1]}')
                #     print(f'trg_prob_weak[i[0],i[1]]:{trg_prob_weak[i[0],i[1]]}')
                # print(f'trg_prob_weak:{trg_prob_weak}')
                # print(f'trg_pred_weak:{trg_pred_weak}')
                # non_candidate_loss = 0
                non_candidate_loss = torch.zeros(1).to(self.device)
                for i in ind_non_candidate:
                    non_candidate_loss += torch.log(1 - trg_prob_weak[i[0],i[1]]).to(self.device)
                    # non_candidate_loss += torch.log(1 - trg_pred_weak[i[0],i[1]])
                    # print(f'trg_pred_weak[{i[0]},{i[1]}]:{trg_pred_weak[i[0],i[1]]}')
                    # print(f'torch.log(1 - trg_prob_weak[{i[0]},{i[1]}]):{torch.log(1 - trg_prob_weak[i[0],i[1]])}')
                    # print(f'torch.log(1 - trg_pred_weak[{i[0]},{i[1]}]):{torch.log(1 - trg_pred_weak[i[0],i[1]])}')
                    # print(f'non_candidate_loss:{non_candidate_loss}')

                non_candidate_loss = -non_candidate_loss/self.hparams["batch_size"]
                # print(f'average -non_candidate_loss:{non_candidate_loss}')
                # w = torch.div(torch.t(conf_sel),num_candidate)
                # w = torch.t(w)
                # w[ind_candidate] = 1/num_candidate
                # print(f'after initialize w:{w}')
                # print(f'len(trg_x):{len(trg_x)}')
                # pseudo_candidate_set = trg_x[ind_candidate]
                # pseudo_labels = pseudo_labels[ind_candidate]
                # print(f'after ind_keep len(trg_x):{len(trg_x)}')
                # ind_keep = torch.squeeze(conf_sel.nonzero(), dim=-1)
                # print(f'conf_sel:{conf_sel}')
                # print(f'ind_keep:{ind_keep}')
                # end select pseudo-labels based on threshold ------------------------------------------
                

                # # CE_loss = self.cross_entropy(outputs_sorted.logits, pseudo_labels_sorted)
                # # CE_loss = self.cross_entropy(output_target[ind_loss_update], pseudo_labels_sorted)
                # CE_loss = self.cross_entropy(trg_pred, pseudo_labels)
                # # print(f'CE_loss:{CE_loss}')
                # # end Lama -----------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                # loss_refine = candidate_loss + self.hparams["lam"] * (non_candidate_loss)
                loss_refine = candidate_loss + self.hparams["lam"] * (non_candidate_loss)
                # loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                # loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')
                

                # self.optimizer.zero_grad()
                # loss_refine.backward()
                # self.optimizer.step()
                # self.optimizerRefine.zero_grad()
                optimizer.zero_grad()
                loss_refine.backward()
                optimizer.step()
                # self.optimizerRefine.step()

                
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                losses_refine = {'candidate_loss': candidate_loss.detach().item(), 'non_candidate_loss': non_candidate_loss.detach().item(), 'Total_loss': loss_refine.detach().item()}
                # losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                #  'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses_refine.items():
                    avg_meter[key].update(val, 32)


            # Update w



            # Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # self.lr_scheduler.step()
            # self.lr_schedulerRefine.step()
            lr_scheduler.step()



            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_iter"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def refine2(self, trg_dataloader, avg_meter, logger, modelmoment, optimizer, lr_scheduler):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = modelmoment.state_dict()
        self.last_model = modelmoment.state_dict()

        total_epochs = self.hparams["num_iter"]

        for epoch in range(total_epochs):
            modelmoment.train()

            for step, (trg_x, trg_y, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                trg_y = trg_y.long().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)

                # Source Model predictions for target data
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits
                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)

                # Target model outputs (original, strong, weak augmentations)
                outputs = modelmoment(trg_x)
                trg_pred = outputs.logits
                trg_prob = nn.Softmax(dim=1)(trg_pred)
                trg_conf, _ = torch.max(trg_prob, 1)

                outputs_strong = modelmoment(trg_x_strong)
                trg_pred_strong = outputs_strong.logits
                trg_prob_strong = nn.Softmax(dim=1)(trg_pred_strong)
                trg_conf_strong, _ = torch.max(trg_prob_strong, 1)

                outputs_weak = modelmoment(trg_x_weak)
                trg_pred_weak = outputs_weak.logits
                trg_prob_weak = nn.Softmax(dim=1)(trg_pred_weak)
                trg_conf_weak, _ = torch.max(trg_prob_weak, 1)

                # Candidate set selection based on threshold tau
                # prob_sel: Boolean mask of shape (batch_size, num_classes)
                prob_sel = trg_prob > self.hparams["tau"]
                # num_candidate: number of candidate classes per sample (shape: [batch_size])
                num_candidate = torch.sum(prob_sel, dim=1)

                batch_size, num_classes = trg_prob.shape

                # Vectorized initialization or update of the label confidence vector w
                if epoch == 0:
                    # Initialize w: assign uniform probability over candidate classes
                    w = torch.zeros_like(trg_prob)
                    # For candidate positions, assign 1/num_candidate per sample using broadcasting
                    w[prob_sel] = 1.0 / num_candidate.view(-1, 1).expand_as(trg_prob)[prob_sel]
                else:
                    # Update w using strong predictions for candidate classes only.
                    # Compute, per sample, the sum of strong predictions for candidate classes:
                    candidate_sum = torch.sum(trg_prob_strong * prob_sel.float(), dim=1)  # shape: [batch_size]
                    # Create a new w vector that will update candidate positions only where candidate_sum > 0
                    w_new = w.clone()
                    update_mask = candidate_sum > 0  # boolean mask for samples to update
                    if update_mask.any():
                        # For samples where update_mask is True, compute new candidate values:
                        # Divide strong predictions by candidate_sum (broadcasted) and zero out non-candidate positions.
                        w_update = (trg_prob_strong[update_mask] / candidate_sum[update_mask].view(-1, 1)) * prob_sel[update_mask].float()
                        w_new[update_mask] = w_update
                    # For samples where candidate_sum == 0, retain the previous w values.
                    w = w_new.detach()

                print(f'w:{w}')
                print(f'trg_y[prob_sel]:{trg_y[prob_sel]}')

                # Candidate loss: select positions where strong confidence is below a threshold
                candidate_mask = (trg_conf_strong < self.hparams["threshold"]) & (trg_conf > self.hparams["tau"])
                # print(f'candidate_mask:{candidate_mask}')
                if candidate_mask.sum() > 0:
                    candidate_loss = self.kl_loss(F.log_softmax(w[candidate_mask], dim=1),
                                                  trg_prob_strong[candidate_mask])
                else:
                    candidate_loss = torch.zeros(1).to(self.device)

                # Non-candidate loss: vectorized computation over positions not in the candidate set
                non_prob_mask = ~prob_sel  # positions outside the pseudo-candidate set
                if non_prob_mask.sum() > 0:
                    non_candidate_loss = - torch.log(1 - trg_prob_weak[non_prob_mask]).sum() / batch_size
                else:
                    non_candidate_loss = torch.zeros(1).to(self.device)

                # Overall loss
                loss_refine = candidate_loss + self.hparams["lam"] * non_candidate_loss

                optimizer.zero_grad()
                loss_refine.backward()
                optimizer.step()

                losses_refine = {'candidate_loss': candidate_loss.detach().item(),
                                 'non_candidate_loss': non_candidate_loss.detach().item(),
                                 'Total_loss': loss_refine.detach().item()}
                for key, val in losses_refine.items():
                    avg_meter[key].update(val, batch_size)

            lr_scheduler.step()

            logger.debug(f'[Epoch : {epoch+1}/{total_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug('-------------------------------------')

        return self.last_model, self.best_model

class B2TSDA_COT_refine(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_COT_refine, self).__init__(configs)

        self.best_model_net1 = True
        self.pretrained_target = False

        self.acc = Accuracy(task="multiclass", num_classes=configs.num_classes).to(device)
        self.f1 = F1Score(task="multiclass", num_classes=configs.num_classes, average="macro").to(device)
        self.auroc = AUROC(task="multiclass", num_classes=configs.num_classes).to(device)  

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        self.modelmoment2 = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment2.init()

        # print(f'self.sourceModel:{self.sourceModel}')

        # self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module
        self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        self.modelmoment2 = self.modelmoment2.module


        self.optimizerRefine = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate_refine"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerRefine2 = torch.optim.Adam(
            self.modelmoment2.parameters(),
            lr=hparams["learning_rate_refine"],
            weight_decay=hparams["weight_decay"]
        )


        # device
        self.device = device
        self.hparams = hparams

        self.lr_schedulerRefine = StepLR(self.optimizerRefine, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerRefine2 = StepLR(self.optimizerRefine2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def refine_(self, trg_dataloader, trg_dataloader_test, avg_meter1, avg_meter2, logger, target_model_dir):

        # defining best and last model
        best_src_risk = float('inf')
        # self.best_model = self.modelmoment.state_dict()
        # self.last_model = self.modelmoment.state_dict()
        self.model1 = self.modelmoment.state_dict() # COT
        self.model2 = self.modelmoment2.state_dict()
        self.target_model_dir = target_model_dir

        # for k, v in self.classifier.named_parameters():
        # freeze both classifier and ood detector
        # for k, v in self.classifier.named_parameters():
        #     v.requires_grad = False
        # for k, v in self.temporal_verifier.named_parameters():
        #     v.requires_grad = False

        # print(f'self.pretrained_source:{self.pretrained_source}')

        if not self.pretrained_target:
            print(f'Load target model..')
            load_target_model_path = target_model_dir + "/checkpoint.pt"
            print(f'target model path:{load_target_model_path}')
            self.modelmoment.load_state_dict(torch.load(load_target_model_path)["model1"])
            self.modelmoment2.load_state_dict(torch.load(load_target_model_path)["model2"])
            for param in self.modelmoment.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')
            for param in self.modelmoment2.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')

        self.model1 = self.refine_model(trg_dataloader, trg_dataloader_test, avg_meter1, logger, self.modelmoment, self.optimizerRefine, self.lr_schedulerRefine)
        self.model2 = self.refine_model(trg_dataloader, trg_dataloader_test, avg_meter2, logger, self.modelmoment2, self.optimizerRefine2, self.lr_schedulerRefine2)

        return self.model1, self.model2

    def refine(self, trg_dataloader, trg_dataloader_test, avg_meter, logger, target_model_dir):

        # defining best and last model
        best_src_risk = float('inf')
        # self.best_model = self.modelmoment.state_dict()
        # self.last_model = self.modelmoment.state_dict()
        self.model1 = self.modelmoment.state_dict() # COT
        self.model2 = self.modelmoment2.state_dict()
        self.target_model_dir = target_model_dir

        # for k, v in self.classifier.named_parameters():
        # freeze both classifier and ood detector
        # for k, v in self.classifier.named_parameters():
        #     v.requires_grad = False
        # for k, v in self.temporal_verifier.named_parameters():
        #     v.requires_grad = False

        # print(f'self.pretrained_source:{self.pretrained_source}')

        if not self.pretrained_target:
            print(f'Load target model..')
            load_target_model_path = target_model_dir + "/checkpoint.pt"
            print(f'target model path:{load_target_model_path}')
            self.modelmoment.load_state_dict(torch.load(load_target_model_path)["model1"])
            self.modelmoment2.load_state_dict(torch.load(load_target_model_path)["model2"])
            for param in self.modelmoment.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')
            for param in self.modelmoment2.parameters():
                if not param.data.is_cuda:
                    # print(f'model_t_all[0] param.data:{param.data}')
                    # print(f'param.data.davice:{param.data.device}')
                    param.data = param.to('cuda')

        total_epochs = self.hparams["num_iter"]

        for epoch in range(total_epochs):
            self.modelmoment.train()
            self.modelmoment2.train()

            for step, (trg_x, trg_y, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                trg_y = trg_y.long().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)
                # if epoch == 0:
                #     self.w = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)

                loss_refine, candidate_loss, non_candidate_loss = self.refine_model(trg_x, trg_x_weak, trg_x_strong, self.modelmoment, self.optimizerRefine, epoch)
                loss_refine2, candidate_loss2, non_candidate_loss2 = self.refine_model(trg_x, trg_x_weak, trg_x_strong, self.modelmoment2, self.optimizerRefine2, epoch)

                losses = {'candidate_loss': candidate_loss.detach().item(),
                         'non_candidate_loss': non_candidate_loss.detach().item(),
                         'Total_loss': loss_refine.detach().item(),
                         'candidate_loss2': candidate_loss2.detach().item(),
                         'non_candidate_loss2': non_candidate_loss2.detach().item(),
                         'Total_loss2': loss_refine2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, self.hparams["batch_size"])

            self.lr_schedulerRefine.step()
            self.lr_schedulerRefine2.step()

            logger.debug(f'[Epoch : {epoch+1}/{total_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug('-------------------------------------')
            self.calc_metrics(trg_dataloader_test)
            logger.debug('-------------------------------------')

        self.model1 = deepcopy(self.modelmoment.state_dict())
        self.model2 = deepcopy(self.modelmoment2.state_dict())

        return self.model1, self.model2

    def refine_model(self, trg_x, trg_x_weak, trg_x_strong, modelmoment, optimizer, epoch):
        # Target model outputs (original, strong, weak augmentations)
        outputs = modelmoment(trg_x)
        trg_pred = outputs.logits
        trg_prob = nn.Softmax(dim=1)(trg_pred)
        trg_conf, _ = torch.max(trg_prob, 1)

        outputs_strong = modelmoment(trg_x_strong)
        trg_pred_strong = outputs_strong.logits
        trg_prob_strong = nn.Softmax(dim=1)(trg_pred_strong)
        trg_conf_strong, _ = torch.max(trg_prob_strong, 1)

        outputs_weak = modelmoment(trg_x_weak)
        trg_pred_weak = outputs_weak.logits
        trg_prob_weak = nn.Softmax(dim=1)(trg_pred_weak)
        trg_conf_weak, _ = torch.max(trg_prob_weak, 1)

        # Candidate set selection based on threshold tau
        # prob_sel: Boolean mask of shape (batch_size, num_classes)
        prob_sel = trg_prob > self.hparams["tau"]
        # num_candidate: number of candidate classes per sample (shape: [batch_size])
        num_candidate = torch.sum(prob_sel, dim=1)

        batch_size, num_classes = trg_prob.shape

        # Vectorized initialization or update of the label confidence vector w
        if epoch == 0:
            # Initialize w: assign uniform probability over candidate classes
            self.w = torch.zeros_like(trg_prob)
            # For candidate positions, assign 1/num_candidate per sample using broadcasting
            self.w[prob_sel] = 1.0 / num_candidate.view(-1, 1).expand_as(trg_prob)[prob_sel]
        else:
            # Update w using strong predictions for candidate classes only.
            # Compute, per sample, the sum of strong predictions for candidate classes:
            candidate_sum = torch.sum(trg_prob_strong * prob_sel.float(), dim=1)  # shape: [batch_size]
            # Create a new w vector that will update candidate positions only where candidate_sum > 0
            w_new = self.w.clone()
            update_mask = candidate_sum > 0  # boolean mask for samples to update
            if update_mask.any():
                # For samples where update_mask is True, compute new candidate values:
                # Divide strong predictions by candidate_sum (broadcasted) and zero out non-candidate positions.
                w_update = (trg_prob_strong[update_mask] / candidate_sum[update_mask].view(-1, 1)) * prob_sel[update_mask].float()
                w_new[update_mask] = w_update
            # For samples where candidate_sum == 0, retain the previous w values.
            self.w = w_new.detach()

        # print(f'w:{w}')
        # print(f'trg_y[prob_sel]:{trg_y[prob_sel]}')

        # Candidate loss: select positions where strong confidence is below a threshold
        candidate_mask = (trg_conf_strong < self.hparams["threshold"]) & (trg_conf > self.hparams["tau"])
        # print(f'candidate_mask:{candidate_mask}')
        if candidate_mask.sum() > 0:
            candidate_loss = self.kl_loss(F.log_softmax(self.w[candidate_mask], dim=1),
                                          trg_prob_strong[candidate_mask])
        else:
            candidate_loss = torch.zeros(1).to(self.device)

        # Non-candidate loss: vectorized computation over positions not in the candidate set
        non_prob_mask = ~prob_sel  # positions outside the pseudo-candidate set
        if non_prob_mask.sum() > 0:
            non_candidate_loss = - torch.log(1 - trg_prob_weak[non_prob_mask]).sum() / batch_size
        else:
            non_candidate_loss = torch.zeros(1).to(self.device)

        # Overall loss
        loss_refine = candidate_loss + self.hparams["lam"] * non_candidate_loss

        optimizer.zero_grad()
        loss_refine.backward()
        optimizer.step()
                
        return loss_refine, candidate_loss, non_candidate_loss

    def refine_model_(self, trg_dataloader, trg_dataloader_test, avg_meter, logger, modelmoment, optimizer, lr_scheduler):

        # defining best and last model
        best_src_risk = float('inf')
        # self.best_model = modelmoment.state_dict()
        self.last_model = modelmoment.state_dict()

        total_epochs = self.hparams["num_iter"]

        for epoch in range(total_epochs):
            modelmoment.train()

            for step, (trg_x, trg_y, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                trg_y = trg_y.long().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)

                # Source Model predictions for target data
                # with torch.no_grad():
                #     outputs_src = self.sourceModel(trg_x)
                #     trg_pred_src = outputs_src.logits
                #     src_prob = nn.Softmax(dim=1)(trg_pred_src)
                #     src_conf, pseudo_labels = torch.max(src_prob, 1)

                # Target model outputs (original, strong, weak augmentations)
                outputs = modelmoment(trg_x)
                trg_pred = outputs.logits
                trg_prob = nn.Softmax(dim=1)(trg_pred)
                trg_conf, _ = torch.max(trg_prob, 1)

                outputs_strong = modelmoment(trg_x_strong)
                trg_pred_strong = outputs_strong.logits
                trg_prob_strong = nn.Softmax(dim=1)(trg_pred_strong)
                trg_conf_strong, _ = torch.max(trg_prob_strong, 1)

                outputs_weak = modelmoment(trg_x_weak)
                trg_pred_weak = outputs_weak.logits
                trg_prob_weak = nn.Softmax(dim=1)(trg_pred_weak)
                trg_conf_weak, _ = torch.max(trg_prob_weak, 1)

                # Candidate set selection based on threshold tau
                # prob_sel: Boolean mask of shape (batch_size, num_classes)
                prob_sel = trg_prob > self.hparams["tau"]
                # num_candidate: number of candidate classes per sample (shape: [batch_size])
                num_candidate = torch.sum(prob_sel, dim=1)

                batch_size, num_classes = trg_prob.shape

                # Vectorized initialization or update of the label confidence vector w
                if epoch == 0:
                    # Initialize w: assign uniform probability over candidate classes
                    w = torch.zeros_like(trg_prob)
                    # For candidate positions, assign 1/num_candidate per sample using broadcasting
                    w[prob_sel] = 1.0 / num_candidate.view(-1, 1).expand_as(trg_prob)[prob_sel]
                else:
                    # Update w using strong predictions for candidate classes only.
                    # Compute, per sample, the sum of strong predictions for candidate classes:
                    candidate_sum = torch.sum(trg_prob_strong * prob_sel.float(), dim=1)  # shape: [batch_size]
                    # Create a new w vector that will update candidate positions only where candidate_sum > 0
                    w_new = w.clone()
                    update_mask = candidate_sum > 0  # boolean mask for samples to update
                    if update_mask.any():
                        # For samples where update_mask is True, compute new candidate values:
                        # Divide strong predictions by candidate_sum (broadcasted) and zero out non-candidate positions.
                        w_update = (trg_prob_strong[update_mask] / candidate_sum[update_mask].view(-1, 1)) * prob_sel[update_mask].float()
                        w_new[update_mask] = w_update
                    # For samples where candidate_sum == 0, retain the previous w values.
                    w = w_new.detach()

                # print(f'w:{w}')
                # print(f'trg_y[prob_sel]:{trg_y[prob_sel]}')

                # Candidate loss: select positions where strong confidence is below a threshold
                candidate_mask = (trg_conf_strong < self.hparams["threshold"]) & (trg_conf > self.hparams["tau"])
                # print(f'candidate_mask:{candidate_mask}')
                if candidate_mask.sum() > 0:
                    candidate_loss = self.kl_loss(F.log_softmax(w[candidate_mask], dim=1),
                                                  trg_prob_strong[candidate_mask])
                else:
                    candidate_loss = torch.zeros(1).to(self.device)

                # Non-candidate loss: vectorized computation over positions not in the candidate set
                non_prob_mask = ~prob_sel  # positions outside the pseudo-candidate set
                if non_prob_mask.sum() > 0:
                    non_candidate_loss = - torch.log(1 - trg_prob_weak[non_prob_mask]).sum() / batch_size
                else:
                    non_candidate_loss = torch.zeros(1).to(self.device)

                # Overall loss
                loss_refine = candidate_loss + self.hparams["lam"] * non_candidate_loss

                optimizer.zero_grad()
                loss_refine.backward()
                optimizer.step()

                losses_refine = {'candidate_loss': candidate_loss.detach().item(),
                                 'non_candidate_loss': non_candidate_loss.detach().item(),
                                 'Total_loss': loss_refine.detach().item()}
                for key, val in losses_refine.items():
                    avg_meter[key].update(val, batch_size)

            lr_scheduler.step()

            logger.debug(f'[Epoch : {epoch+1}/{total_epochs}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug('-------------------------------------')
            self.calc_metrics(trg_dataloader_test)

        self.last_model = deepcopy(modelmoment.state_dict())
        return self.last_model

    def eval_aggr(self, test_loader):
        # feature_extractor = self.algorithm.feature_extractor.to(self.device)
        # classifier = self.algorithm.classifier.to(self.device)

        # feature_extractor.eval()
        # classifier.eval()
        # print(f'eval aggr')

        model = self.modelmoment
        model2 = self.modelmoment2
        model.eval()
        model2.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                # features, seq_features = feature_extractor(data)
                # predictions = classifier(features)
                outputs = model(data)
                predictions = outputs.logits
                outputs2 = model2(data)
                predictions2 = outputs2.logits

                max_output, _ = torch.max(predictions, 1)
                max_output2, _ = torch.max(predictions2, 1)

                a = max_output/(max_output+max_output2)
                b = max_output2/(max_output+max_output2)

                # Aggregate output 
                output_target = torch.unsqueeze(a,1)*predictions + torch.unsqueeze(b,1)*predictions2

                # compute loss
                loss = F.cross_entropy(output_target, labels)
                total_loss.append(loss.item())
                pred = output_target.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss_eval = torch.tensor(total_loss).mean()  # average loss
        self.full_preds_eval = torch.cat((preds_list))
        self.full_labels_eval = torch.cat((labels_list))
        # print(f'self.full_preds_eval:{self.full_preds_eval}')

    def eval_single(self, test_loader):
        # feature_extractor = self.algorithm.feature_extractor.to(self.device)
        # classifier = self.algorithm.classifier.to(self.device)

        # feature_extractor.eval()
        # classifier.eval()
        # print(f'eval aggr')

        model = self.modelmoment
        # model2 = self.modelmoment2
        model.eval()
        # model2.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                # features, seq_features = feature_extractor(data)
                # predictions = classifier(features)
                outputs = model(data)
                predictions = outputs.logits
                # outputs2 = model2(data)
                # predictions2 = outputs2.logits

                # max_output, _ = torch.max(predictions, 1)
                # max_output2, _ = torch.max(predictions2, 1)

                # a = max_output/(max_output+max_output2)
                # b = max_output2/(max_output+max_output2)

                # Aggregate output 
                # output = torch.unsqueeze(a,1)*predictions + torch.unsqueeze(b,1)*predictions2

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss_eval = torch.tensor(total_loss).mean()  # average loss
        self.full_preds_eval = torch.cat((preds_list))
        self.full_labels_eval = torch.cat((labels_list))

    def calc_metrics(self, test_loader):
       
        self.eval_aggr(test_loader)
        # self.eval_single(test_loader)
        # self.evaluate_ori(self.trg_test_dl) # 5 Des

        # self.evaluate(self.trg_test_dl)
        # accuracy  
        acc = self.acc(self.full_preds_eval.argmax(dim=1), self.full_labels_eval).item()
        # f1
        f1 = self.f1(self.full_preds_eval.argmax(dim=1), self.full_labels_eval).item()
        # auroc 
        auroc = self.auroc(self.full_preds_eval, self.full_labels_eval).item()

        # print(f'acc\t:{acc}')
        print(f'f1\t:{f1}')
        # print(f'auroc\t:{auroc}')

        return acc, f1, auroc


class B2TSDA_Only(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_Only, self).__init__(configs)

        # self.feature_extractor = backbone(configs)
        # self.classifier = classifier(configs)
        # self.temporal_verifier = Temporal_Imputer(configs)

        # self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        # self.AE_cls = nn.DataParallel(self.AE_cls)
        # self.AE_cls = self.AE_cls.module

        # self.AE_cls2 = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        # self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        # self.AE_cls2 = self.AE_cls2.module

        self.best_model_net1 = True
        self.pretrained_source = False

        self.sourceModel = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.sourceModel.init()

        self.modelmoment = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.configs.input_channels,
                "num_class": self.configs.num_classes,
                "sequence_len": self.configs.sequence_len,
                "num_layer": self.configs.prompt_length,
                "prompt_init": "uniform",
                "dropout": configs.dropout,
            },
        )
        self.modelmoment.init()

        # self.modelmoment2 = MOMENTPipeline.from_pretrained(
        #     "AutonLab/MOMENT-1-large", 
        #     model_kwargs={
        #         "task_name": "classification",
        #         "n_channels": self.configs.input_channels,
        #         "num_class": self.configs.num_classes,
        #         "sequence_len": self.configs.sequence_len,
        #         "num_layer": self.configs.prompt_length,
        #         "prompt_init": "uniform",
        #         "dropout": configs.dropout,
        #     },
        # )
        # self.modelmoment2.init()

        # print(f'self.sourceModel:{self.sourceModel}')

        # self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.sourceModel = nn.DataParallel(self.sourceModel)
        self.sourceModel = self.sourceModel.module
        self.modelmoment = nn.DataParallel(self.modelmoment)
        self.modelmoment = self.modelmoment.module
        # self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        # self.modelmoment2 = self.modelmoment2.module
        # self.AE_cls = nn.DataParallel(self.AE_cls)
        # self.AE_cls = self.AE_cls.module
        # self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        # self.AE_cls2 = self.AE_cls2.module


        self.optimizer = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizerRefine = torch.optim.Adam(
            self.modelmoment.parameters(),
            lr=hparams["learning_rate_refine"],
            weight_decay=hparams["weight_decay"]
        )

        # self.optimizer2 = torch.optim.Adam(
        #     self.modelmoment2.parameters(),
        #     lr=hparams["learning_rate"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.optimizerAE = torch.optim.Adam(
        #     self.AE_cls.parameters(),
        #     lr=hparams["learning_rate_AE"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.optimizerAE2 = torch.optim.Adam(
        #     self.AE_cls2.parameters(),
        #     lr=hparams["learning_rate_AE"],
        #     weight_decay=hparams["weight_decay"]
        # )

        self.pre_optimizer = torch.optim.Adam(
            self.sourceModel.parameters(),
            # self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_schedulerRefine = StepLR(self.optimizerRefine, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _, _, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                # print(f'src_x:{src_x}')

                # forward pass correct sequences
                outputs = self.sourceModel(src_x)
                # src_feat, _ = self.feature_extractor(src_x)


                # classifier predictions
                src_pred = outputs.logits
                # src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.sourceModel.state_dict())
        return src_only_model

    # def update(self, trg_dataloader, avg_meter, logger, source_model_dir):
    # def update(self, trg_dataloader, avg_meter, logger):
    def update(self, trg_dataloader, trg_test_dataloader, avg_meter, logger, num_neighbors):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = self.modelmoment.state_dict()
        self.last_model = self.modelmoment.state_dict()
        # self.source_model_dir = source_model_dir

        # for k, v in self.classifier.named_parameters():
        # freeze both classifier and ood detector
        # for k, v in self.classifier.named_parameters():
        #     v.requires_grad = False
        # for k, v in self.temporal_verifier.named_parameters():
        #     v.requires_grad = False

        # print(f'self.pretrained_source:{self.pretrained_source}')

        # if not self.pretrained_source:
        #     print(f'Load pretrained source model..')
        #     load_source_model_path = source_model_dir + "/checkpoint.pt"
        #     print(f'source model path:{load_source_model_path}')
        #     self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
        #     for param in self.sourceModel.parameters():
        #         if not param.data.is_cuda:
        #             # print(f'model_t_all[0] param.data:{param.data}')
        #             # print(f'param.data.davice:{param.data.device}')
        #             param.data = param.to('cuda')

        total_epochs = self.hparams["num_epochs"] 
        # forget_rate = np.ones(total_epochs) * self.hparams["forget_rate"]
        # forget_rate[:(self.hparams["warm_target"]+self.hparams["num_gradual"])] = np.linspace(0, self.hparams["forget_rate"], (self.hparams["warm_target"]+self.hparams["num_gradual"]))
        # rate_schedule = np.ones(total_epochs) * self.hparams["forget_rate"]
        # rate_schedule[:(self.hparams["num_gradual"])] = np.linspace(0, self.hparams["forget_rate"]**self.hparams["exponent"], self.hparams["num_gradual"])
        # print(f'rate_schedule:{rate_schedule}')
        # print(f'rate_schedule.shape:{rate_schedule.shape}')
        # banks = eval_and_label_dataset(0, self.feature_extractor, self.classifier_t, None, trg_test_dataloader, trg_dataloader, num_neighbors)
        
        # banks, acc = eval_and_label_dataset(0, self.sourceModel, self.modelmoment, None, trg_test_dataloader, num_neighbors)
        # print(f'acc:{acc}')

        # train
        for epoch in range(0, total_epochs):
        # for epoch in range(1, self.hparams["num_epochs"] + 1):
            # if epoch <= round(total_epochs/2):#total_epochs
            #     alpha = 0.0
            # else:
            #     alpha = (epoch*2-total_epochs)/total_epochs

            
            self.modelmoment.train()
            # self.modelmoment2.train()

            for step, (trg_x, _, trg_idx, trg_x_weak, _) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)

                # Validation data
                # val_x, val_y, _ = next(iter(val_dataloader))
                # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    # outputs_src = self.sourceModel(trg_x_weak)
                    trg_pred_src = outputs_src.logits
                    # trg_feat_src = outputs_src.embeddings2

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)

                # # Pseudo-label refinement
                # with torch.no_grad():
                #     # probs = F.softmax(trg_pred, dim=1)
                #     probs = src_prob
                #     # probs = probs_out_std
                #     # probs2 = F.softmax(out_un, dim=1)
                #     pseudo_labels, probs_refine, _, _ = refine_predictions(trg_feat_src, probs, banks, num_neighbors)
                #     # print(f'pseudo_labels:{pseudo_labels}')
                #     # print(f'pseudo_labels.shape:{pseudo_labels.shape}')
                

                # Co-teaching model output
                # with torch.no_grad():
                #     outputs_ind = self.modelmoment2(trg_x)
                #     trg_pred_ind = outputs_ind.logits
                #     trg_ind_prob = nn.Softmax(dim=1)(trg_pred_ind)

                # Target Model1 output
                outputs = self.modelmoment(trg_x)
                # outputs = self.modelmoment(trg_x_weak)
                trg_pred = outputs.logits
                # trg_feat = outputs.embeddings
                # trg_feat = outputs.embeddings2
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # # Pseudo-label refinement --------------------------------------------------
                # with torch.no_grad():
                #     # probs = F.softmax(trg_pred, dim=1)
                #     probs = src_prob
                #     # probs = probs_out_std
                #     # probs2 = F.softmax(out_un, dim=1)
                #     pseudo_labels, probs_refine, _, _ = refine_predictions(trg_feat, probs, banks, num_neighbors)
                #     # print(f'pseudo_labels:{pseudo_labels}')
                #     # print(f'pseudo_labels.shape:{pseudo_labels.shape}')
                # # ---------------------------------------------------------------------------

                # Target Model2 output
                # outputs2 = self.modelmoment2(trg_x)
                # trg_pred2 = outputs2.logits

                # co-teaching loss
                # loss_1, loss_2 = self.loss_coteaching(trg_pred, trg_pred2, pseudo_labels, rate_schedule[epoch], trg_idx)

                # reconstruct -------------------------------------------------------------
                mt = random.uniform(0,1) #mask
                s0,s1,s2 = trg_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                mt = torch.bernoulli(randuniform).to(self.device)
                m_ones = torch.ones(s0,s1,s2).to(self.device)

                sum_mt = mt.flatten().sum()
                # c_mask = 1/((s0*s1*s2) * sum_mt)
                # c_unmask = 1/(((s0*s1*s2) - sum_mt)*(s0*s1*s2))
                c_mask = sum_mt/(s0*s1*s2)
                c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)
                # print(f'c_mask:{c_mask}')
                # print(f'c_unmask:{c_unmask}')
                # print(f'mt:{mt}')
                # print(f'trg_x.shape:{trg_x.shape}')

                #src if mt=1 -> x=0
                src2 = torch.clone(trg_x)
                src2 = src2 * (m_ones-mt)
                gamma = 0.5
                criterion = RMSELoss()
                # print(f'src2.shape:{src2.shape}')

                # pred reconstruct
                out = self.modelmoment.forward_reconstruct(src2)
                pred_reconstruct = out.reconstruction

                # out2 = self.modelmoment2.forward_reconstruct(src2)
                # pred_reconstruct2 = out2.reconstruction
                # print(f'pred_reconstruct.shape:{pred_reconstruct.shape}')
                # print(f'src2.shape:{src2.shape}')
                # print(f'criterion(pred_reconstruct, src2):{criterion(pred_reconstruct, src2)}')
                #loss reconstruct
                loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)
                # loss_reconstruct2 = gamma * c_mask * criterion(pred_reconstruct2, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct2, src2)
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # -------------------------------------------------------------------------

                
                # # Lama -------------------------------------------------------------------------
                # loss_cot = F.cross_entropy(trg_pred_ind, pseudo_labels, reduction='none') # reduction='none' to get loss per batch instead of avg loss
                # # loss_cot = F.cross_entropy(output_target, pseudo_labels, reduction='none')
                # # print(f'loss_cot:{loss_cot}')
                
                # # get (1-R) percent low loss samples
                # ind_loss_sorted = np.argsort(loss_cot.cpu().data).to(self.device)
                # # ind_loss_sorted_thres = np.argsort(trg_ind_prob.cpu().data).to(self.device)

                # remember_rate = 1 - (1-alpha) * forget_rate
                # num_remember = math.ceil(remember_rate * len(ind_loss_sorted))
                # ind_loss_update = ind_loss_sorted[:num_remember]
                # # ind_loss_neg_update = ind_loss_sorted[(num_remember):]#/*2+num_neg

                # # print(f'remember_rate:{remember_rate}')
                # # print(f'len(ind_loss_update):{len(ind_loss_update)}')
                # # print(f'num data:{num_remember}')
                # # print(f'trg_x[ind_loss_update]:{trg_x[ind_loss_update]}')
                # # outputs_sorted = modelmoment(trg_x[ind_loss_update])
                # # trg_pred2 = outputs_sorted.logits

                # # outputs_ind_sorted = modelmoment_ind(trg_x[ind_loss_update])
                # # trg_pred_ind2 = outputs_ind_sorted.logits

                # # max_output2, _ = torch.max(trg_pred2, 1)
                # # max_output_ind2, _ = torch.max(trg_pred_ind2, 1)

                # # a2 = max_output2/(max_output2+max_output_ind2)
                # # b2 = max_output_ind2/(max_output2+max_output_ind2)
                # # output_target2 = torch.unsqueeze(a2,1)*trg_pred2 + torch.unsqueeze(b2,1)*trg_pred_ind2

                
                # # trg_prob_sorted = torch.log_softmax(outputs_sorted.logits, dim=1)

                # # with torch.no_grad():
                # #     outputs_src_sorted = sourceModel(trg_x[ind_loss_update])
                # #     trg_pred_src_sorted = outputs_src_sorted.logits

                # # trg_prob_src_sorted = nn.Softmax(dim=1)(trg_pred_src_sorted)
                # # _, pseudo_labels_sorted = torch.max(trg_prob_src_sorted, 1)

                # pseudo_labels_sorted = pseudo_labels[ind_loss_update].to(self.device)

                # # with torch.no_grad():
                # #     outputs_src2 = sourceModel(trg_x[ind_loss_update])
                # #     trg_pred_src2 = outputs_src2.logits

                # # src_prob2 = nn.Softmax(dim=1)(trg_pred_src2)
                # # _, pseudo_labels2 = torch.max(src_prob2, 1)
                # # print(f'pseudo_labels_sorted:{pseudo_labels_sorted}')
                # # print(f'pseudo_labels2:{pseudo_labels2}')
                # # print(f'trg_prob_sorted:{trg_prob_sorted}')
                # # print(f'prompt.shape:{prompt.shape}')

                # # Entropy loss
                # # trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
                # # print(f'torch.mean(EntropyLoss(trg_prob_sorted):{torch.mean(EntropyLoss(trg_prob_sorted))}')
                # # trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob_sorted))

                # # trg_prob_sorted = torch.log_softmax(trg_pred[ind_loss_update], dim=1).cuda()
                # # print(f'trg_prob_sorted:{trg_prob_sorted}')
                # # print(f'trg_prob_sorted.shape:{trg_prob_sorted.shape}')
                

                # # CE_loss = self.cross_entropy(outputs_sorted.logits, pseudo_labels_sorted)
                # # CE_loss = self.cross_entropy(output_target[ind_loss_update], pseudo_labels_sorted)
                CE_loss = self.cross_entropy(trg_pred, pseudo_labels)
                # # print(f'CE_loss:{CE_loss}')
                # # end Lama -----------------------------------------------------------------------------------

                # # prompt reconstruction -------------------------------------------------------------------------
                # AE_loss = 0
                # # AE_loss2 = 0
                # # use prompt
                # prompt = self.modelmoment.prompt.prompt
                # prompt_new = self.AE_cls(prompt)
                # # prompt2 = self.modelmoment2.prompt.prompt
                # # prompt_new2 = self.AE_cls2(prompt2)

                # AE_loss = torch.pow(torch.linalg.norm(prompt_new - prompt)/prompt.shape[0], 2)
                # # AE_loss2 = torch.pow(torch.linalg.norm(prompt_new2 - prompt2)/prompt2.shape[0], 2)
                # # print(f'AE_loss:{AE_loss}')

                # while AE_loss > 5:
                #     AE_loss = AE_loss / 10

                # # while AE_loss2 > 5:
                # #     AE_loss2 = AE_loss2 / 10
                
                # # -----------------------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                # loss = CE_loss + AE_loss + loss_reconstruct
                # loss = CE_loss
                loss = CE_loss + loss_reconstruct 
                # loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                # loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')

                # update_labels(banks, trg_idx, trg_feat, trg_pred_src)
                

                self.optimizer.zero_grad()
                # self.optimizerAE.zero_grad()
                loss.backward()
                self.optimizer.step()

                # self.optimizer2.zero_grad()
                # loss2.backward()
                # self.optimizer2.step()

                
                # self.optimizerAE.step()
                # self.optimizerAE2.step()
                # self.decoder_optimizer.step()
                # self.global_step+=1


                losses = {'entropy_loss': CE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                #  'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)


            # Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            

            self.lr_scheduler.step()
            # self.lr_scheduler2.step()
            # self.lr_schedulerAE.step()
            # self.lr_schedulerAE2.step()

            # # saving the best model based on src risk
            # if (epoch + 1) % 10 == 0 and (loss_1.avg < best_src_risk or loss_2.avg < best_src_risk):
            #     # best_src_risk = avg_meter['Total_loss'].avg

            #     if loss_1.avg < loss_2.avg:
            #         best_src_risk = loss_1.avg
            #         self.best_model = deepcopy(self.modelmoment.state_dict())
            #         self.best_model_net1 = True
            #     else:
            #         best_src_risk = loss_2.avg
            #         self.best_model = deepcopy(self.modelmoment2.state_dict())
            #         self.best_model_net1 = False


            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def refine(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = self.modelmoment.state_dict()
        self.last_model = self.modelmoment.state_dict()
        # self.source_model_dir = source_model_dir

        total_epochs = self.hparams["num_iter"] 
        # w = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)
        # print(f'w.shape:{w.shape}')

        # train
        for epoch in range(0, total_epochs):
        # for epoch in range(1, self.hparams["num_epochs"] + 1):

            # non_candidate_loss = 0
            self.modelmoment.train()

            for step, (trg_x, _, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)

                # Validation data
                # val_x, val_y, _ = next(iter(val_dataloader))
                # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)


                # Target Model output
                outputs = self.modelmoment(trg_x)
                trg_pred = outputs.logits

                trg_prob = nn.Softmax(dim=1)(trg_pred)
                trg_conf, _ = torch.max(trg_prob, 1)
                # print(f'trg_prob:{trg_prob}')


                # Target Model output with strong aug
                outputs_strong = self.modelmoment(trg_x_strong)
                trg_pred_strong = outputs_strong.logits

                trg_prob_strong = nn.Softmax(dim=1)(trg_pred_strong)
                trg_conf_strong, _ = torch.max(trg_prob_strong, 1)

                # Target Model output with weak aug
                outputs_weak = self.modelmoment(trg_x_weak)
                trg_pred_weak = outputs_weak.logits

                trg_prob_weak = nn.Softmax(dim=1)(trg_pred_weak)
                trg_conf_weak, _ = torch.max(trg_prob_weak, 1)
                # print(f'trg_prob_weak:{trg_prob_weak}')

                # print(f'trg_x_strong.shape:{trg_x_strong.shape}')
                # print(f'trg_x_strong:{trg_x_strong}')
                # print(f'trg_pred_strong:{trg_pred_strong}')
                # print(f'trg_prob_strong:{trg_prob_strong}')


                # select pseudo-candidate set (Z) based on threshold -----------------------------------------------
                prob_sel = trg_prob > self.hparams["tau"] 
                non_prob_sel = ~prob_sel              
                # calculate norm |Z|
                num_candidate = torch.sum(prob_sel, dim=1)
                # print(f'num_candidate:{num_candidate}')

                # select samples based on threshold
                conf_sel = trg_conf > self.hparams["tau"]
                non_conf_sel = ~conf_sel


                # print(f'prob_sel:{prob_sel}')
                # print(f'non_prob_sel:{non_prob_sel}')
                # print(f'conf_sel:{conf_sel}')
                # print(f'len(conf_sel):{len(conf_sel)}')
                # print(f'range(len(conf_sel)):{range(len(conf_sel))}')
                # print(f'~conf_sel:{~conf_sel}')
                # print(f'trg_conf.shape:{trg_conf.shape}')

                ind_candidate = torch.argwhere(prob_sel)
                ind_non_candidate = torch.argwhere(non_prob_sel)

                ind_cand_sampl = torch.argwhere(conf_sel)
                ind_non_cand_sampl = torch.argwhere(non_conf_sel)
                
                # print(f'ind_candidate:{ind_candidate}')
                # print(f'ind_cand_sampl:{ind_cand_sampl}')
                # print(f'ind_non_candidate:{ind_non_candidate}')

                w = torch.zeros(trg_prob.shape).to(self.device)
                # print(f'w:{w}')
                probs_strong = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'trg_prob_strong[0]:{trg_prob_strong[0]}')

                # initialized w
                if epoch == 0:
                    for i in ind_candidate:
                        w[i[0],i[1]] = 1/num_candidate[i[0]]
                        # print(f'i:{i}')
                        # print(f'num_candidate[i]:{num_candidate[i]}')
                else:
                    for i in ind_candidate:
                        probs_strong[i[0],i[1]] = trg_prob_strong[i[0],i[1]]

                    for i in ind_candidate:
                        w[i[0],i[1]] = trg_prob_strong[i[0],i[1]]/torch.sum(trg_prob_strong[i[0]])

                # candidate_sel = trg_conf_strong[ind_cand_sampl] < self.hparams["threshold"]
                # print(f'after intialized w:{w}')

                candidate_sel = trg_conf_strong < self.hparams["threshold"]

                ind_candidate_sel = torch.logical_and(conf_sel, candidate_sel)
                ind_candidate_loss = torch.squeeze(ind_candidate_sel.nonzero(), dim=-1)
                # print(f'ind_candidate_loss2:{ind_candidate_loss2}')
                # print(f'ind_keep:{ind_keep}')

                # print(f'trg_conf_strong:{trg_conf_strong}')
                # print(f'trg_conf:{trg_conf}')
                # print(f'trg_conf_strong[ind_cand_sampl]:{trg_conf_strong[ind_cand_sampl]}')
                # print(f'trg_conf_strong[ind_candidate]:{trg_conf_strong[ind_candidate]}')
                # candidate_sel = trg_conf_strong[ind_candidate] < self.hparams["threshold"]
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'trg_prob_strong[ind_candidate]:{trg_prob_strong[ind_candidate]}')
                # print(f'trg_conf_strong:{trg_conf_strong}')
                # print(f'candidate_sel:{candidate_sel}')
                # if epoch > 0:
                # print(f'trg_x_strong:{trg_x_strong}')
                # print(f'ind_candidate:{ind_candidate}')
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'probs_strong:{probs_strong}')
                # print(f'w:{w}')

                # ind_candidate_loss = torch.argwhere(candidate_sel)
                # ind_candidate_loss = torch.argwhere(torch.squeeze(candidate_sel, dim=-1))
                # print(f'ind_candidate_loss:{ind_candidate_loss}')
                # print(f'w[ind_candidate_loss]:{w[ind_candidate_loss]}')
                # print(f'trg_prob_strong[ind_candidate_loss]:{trg_prob_strong[ind_candidate_loss]}')

                # print(f'w[ind_candidate_loss]:{w[ind_candidate_loss]}')
                # print(f'trg_prob_strong[ind_candidate_loss].shape:{trg_prob_strong[ind_candidate_loss].shape}')

                # log_w = np.log(w.cpu())
                # print(f'w[torch.squeeze(ind_candidate_loss, dim=-1)]:{w[torch.squeeze(ind_candidate_loss, dim=-1)]}')
                # print(f'w:{w}')
                # print(f'w[ind_candidate_loss].shape:{w[ind_candidate_loss].shape}')
                # print(f'F.log_softmax(w[ind_candidate_loss]):{F.log_softmax(w[ind_candidate_loss], dim=1)}')
                # print(f'nn.Softmax(w[ind_candidate_loss]):{nn.Softmax(dim=-1)(w[ind_candidate_loss])}')
                if ind_candidate_loss.numel():
                    # print(f'ind_candidate_loss:{ind_candidate_loss}')
                    # print(f'w[ind_candidate_loss].shape:{w[ind_candidate_loss].shape}')
                    # print(f'trg_prob_strong[ind_candidate_loss].shape:{trg_prob_strong[ind_candidate_loss].shape}')
                    # print(f'w[torch.squeeze(ind_candidate_loss, dim=-1)].shape:{w[torch.squeeze(ind_candidate_loss, dim=-1)].shape}')
                    # print(f'trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)].shape:{trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)].shape}')
                    # candidate_loss = self.kl_loss(F.log_softmax(w[torch.squeeze(ind_candidate_loss, dim=-1)], dim=1), trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)])
                    candidate_loss = self.kl_loss(F.log_softmax(w[ind_candidate_loss], dim=1), trg_prob_strong[ind_candidate_loss])
                else:
                    candidate_loss = torch.zeros(1).to(self.device)
                    # candidate_loss = 0.0
                # candidate_loss = self.kl_loss(F.log_softmax(w, dim=1), trg_prob_strong)
                # candidate_loss = self.kl_loss(trg_prob_strong[ind_candidate_loss], F.log_softmax(w[ind_candidate_loss], -1))
                # candidate_loss = self.kl_loss(nn.Softmax(dim=-1)(w[ind_candidate_loss]), trg_prob_strong[ind_candidate_loss])
                # candidate_loss = self.kl_loss(trg_prob_strong[ind_candidate_loss], w[ind_candidate_loss])
                # print(f'candidate_loss:{candidate_loss}')
                # print(f'ind_non_candidate:{ind_non_candidate}')

                # print(f'trg_prob_weak[ind_non_candidate]:{trg_prob_weak[ind_non_candidate]}')

                # for i in ind_non_candidate:
                #     # print(f'i[0]:{i[0]}')
                #     # print(f'i[1]:{i[1]}')
                #     print(f'trg_prob_weak[i[0],i[1]]:{trg_prob_weak[i[0],i[1]]}')
                # print(f'trg_prob_weak:{trg_prob_weak}')
                # print(f'trg_pred_weak:{trg_pred_weak}')
                # non_candidate_loss = 0
                non_candidate_loss = torch.zeros(1).to(self.device)
                for i in ind_non_candidate:
                    non_candidate_loss += torch.log(1 - trg_prob_weak[i[0],i[1]]).to(self.device)
                    # non_candidate_loss += torch.log(1 - trg_pred_weak[i[0],i[1]])
                    # print(f'trg_pred_weak[{i[0]},{i[1]}]:{trg_pred_weak[i[0],i[1]]}')
                    # print(f'torch.log(1 - trg_prob_weak[{i[0]},{i[1]}]):{torch.log(1 - trg_prob_weak[i[0],i[1]])}')
                    # print(f'torch.log(1 - trg_pred_weak[{i[0]},{i[1]}]):{torch.log(1 - trg_pred_weak[i[0],i[1]])}')
                    # print(f'non_candidate_loss:{non_candidate_loss}')

                non_candidate_loss = -non_candidate_loss/self.hparams["batch_size"]
                # print(f'average -non_candidate_loss:{non_candidate_loss}')
                # w = torch.div(torch.t(conf_sel),num_candidate)
                # w = torch.t(w)
                # w[ind_candidate] = 1/num_candidate
                # print(f'after initialize w:{w}')
                # print(f'len(trg_x):{len(trg_x)}')
                # pseudo_candidate_set = trg_x[ind_candidate]
                # pseudo_labels = pseudo_labels[ind_candidate]
                # print(f'after ind_keep len(trg_x):{len(trg_x)}')
                # ind_keep = torch.squeeze(conf_sel.nonzero(), dim=-1)
                # print(f'conf_sel:{conf_sel}')
                # print(f'ind_keep:{ind_keep}')
                # end select pseudo-labels based on threshold ------------------------------------------
                

                # # CE_loss = self.cross_entropy(outputs_sorted.logits, pseudo_labels_sorted)
                # # CE_loss = self.cross_entropy(output_target[ind_loss_update], pseudo_labels_sorted)
                # CE_loss = self.cross_entropy(trg_pred, pseudo_labels)
                # # print(f'CE_loss:{CE_loss}')
                # # end Lama -----------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                # loss_refine = candidate_loss + self.hparams["lam"] * (non_candidate_loss)
                loss_refine = candidate_loss + self.hparams["lam"] * (non_candidate_loss)
                # loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                # loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')
                

                # self.optimizer.zero_grad()
                # loss_refine.backward()
                # self.optimizer.step()
                self.optimizerRefine.zero_grad()
                loss_refine.backward()
                self.optimizerRefine.step()

                
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                losses_refine = {'candidate_loss': candidate_loss.detach().item(), 'non_candidate_loss': non_candidate_loss.detach().item(), 'Total_loss': loss_refine.detach().item()}
                # losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                #  'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses_refine.items():
                    avg_meter[key].update(val, 32)


            # Update w



            # Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # self.lr_scheduler.step()
            self.lr_schedulerRefine.step()



            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_iter"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

class MAPU2(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU2, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.feature_extractor_tgt = backbone(configs)
        self.classifier_tgt = classifier(configs)
        self.temporal_verifier_tgt = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.network_tgt = nn.Sequential(self.feature_extractor_tgt, self.classifier_tgt)

        self.optimizer = torch.optim.Adam(
            self.network_tgt.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer_tgt = torch.optim.Adam(
            self.temporal_verifier_tgt.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _, _, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network_tgt.state_dict()
        last_model = self.network_tgt.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        self.feature_extractor.eval()
        self.classifier.eval()

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx, _, _) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.tov_optimizer_tgt.zero_grad()

                #pseudo-label
                with torch.no_grad():
                    src_feat, src_feat_seq = self.feature_extractor(trg_x)
                    # masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                    # src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                    # src_tov_predictions = self.temporal_verifier(seq_src_feat_mask)
                    # src_tov_loss = self.mse_loss(src_tov_predictions, src_feat_seq)

                    # prediction scores
                    src_pred = self.classifier(src_feat)
                    src_prob = nn.Softmax(dim=1)(src_pred)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)

                    # outputs_src = self.sourceModel(trg_x)
                    # # outputs_src = self.sourceModel(trg_x_weak)
                    # trg_pred_src = outputs_src.logits
                    # src_feat = outputs_src.embeddings
                    # # trg_feat_src = outputs_src.embeddings2

                    # src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    # src_conf, pseudo_labels = torch.max(src_prob, 1)

                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor_tgt(trg_x)
                print(f'trg_feat.shape:{trg_feat.shape}')

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor_tgt(masked_data)

                tov_predictions = self.temporal_verifier_tgt(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier_tgt(trg_feat)
                print(f'trg_pred.shape:{trg_pred.shape}')

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                # reconstruct -------------------------------------------------------------
                mt = random.uniform(0,1) #mask
                s0,s1,s2 = trg_x.shape
                randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                mt = torch.bernoulli(randuniform).to(self.device)
                m_ones = torch.ones(s0,s1,s2).to(self.device)

                sum_mt = mt.flatten().sum()
                # c_mask = 1/((s0*s1*s2) * sum_mt)
                # c_unmask = 1/(((s0*s1*s2) - sum_mt)*(s0*s1*s2))
                c_mask = sum_mt/(s0*s1*s2)
                c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)
                # print(f'c_mask:{c_mask}')
                # print(f'c_unmask:{c_unmask}')
                # print(f'mt:{mt}')
                # print(f'trg_x.shape:{trg_x.shape}')

                #src if mt=1 -> x=0
                src2 = torch.clone(trg_x)
                src2 = src2 * (m_ones-mt)
                gamma = 0.5
                criterion = RMSELoss()
                # print(f'src2.shape:{src2.shape}')

                # pred reconstruct
                out = self.modelmoment.forward_reconstruct(src2)
                pred_reconstruct = out.reconstruction

                # out2 = self.modelmoment2.forward_reconstruct(src2)
                # pred_reconstruct2 = out2.reconstruction
                # print(f'pred_reconstruct.shape:{pred_reconstruct.shape}')
                # print(f'src2.shape:{src2.shape}')
                # print(f'criterion(pred_reconstruct, src2):{criterion(pred_reconstruct, src2)}')
                #loss reconstruct
                loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)
                # loss_reconstruct2 = gamma * c_mask * criterion(pred_reconstruct2, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct2, src2)
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # -------------------------------------------------------------------------

                CE_loss = self.cross_entropy(trg_pred, pseudo_labels)
                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + self.hparams['TOV_wt'] * tov_loss + loss_reconstruct
                loss = CE_loss + self.hparams['TOV_wt'] * tov_loss + loss_reconstruct

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer_tgt.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class B2TSDA_TimeCLR(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(B2TSDA_TimeCLR, self).__init__(configs)

        # self.feature_extractor = backbone(configs)
        # self.classifier = classifier(configs)
        # self.temporal_verifier = Temporal_Imputer(configs)
        # print(f'self.configs.dropout_src:{self.configs.dropout_src}')
        # print(f'configs:{configs}')
        self.encoder_src = backbone(configs, self.configs.dropout_src)
        self.encoder_trg = backbone(configs, self.configs.dropout)
        freeze=['in_net', 'transformer', 'out_net', 'projector', 'dummy', 'start_token']
        
        for n, p in self.encoder_src.named_parameters():
            if n.startswith(tuple(freeze)):
                p.requires_grad = False
        for n, p in self.encoder_trg.named_parameters():
            if n.startswith(tuple(freeze)):
                p.requires_grad = False
        # self.encoder_src = backbone()
        # self.encoder_trg = backbone()
        # self.backbone = Transformer()

        # pre_train_model_path = '/home/furqon/blackbox/foundation/mapu4/MAPU_SFDA_TS/models/trf_tc_0000_0399.npz'
        # pkl = torch.load(pre_train_model_path, map_location='cpu')
        # self.encoder.load_state_dict(pkl['model_state_dict'])
        
        # self.encoder = self.get_timeclr(self.backbone)
        self.encoder_src = self.get_timeclr(self.encoder_src)
        self.encoder_trg = self.get_timeclr(self.encoder_trg)
        # print(f'self.encoder:{self.encoder}')
        # encoder = self.encoder
        self.model_src = Classifier(self.encoder_src, self.configs.num_classes)
        self.model_trg = Classifier(self.encoder_trg, self.configs.num_classes)
        # print(f'self.classifier.encoder:{self.classifier.encoder}')
        # print(f'self.model_src:{self.model_src}')

        # self.AE_cls = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        # self.AE_cls = nn.DataParallel(self.AE_cls)
        # self.AE_cls = self.AE_cls.module

        # self.AE_cls2 = AutoEncoder(1024, configs.mid_dim, configs.out_dim).to(device)
        # self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        # self.AE_cls2 = self.AE_cls2.module

        self.best_model_net1 = True
        self.pretrained_source = False

        # self.sourceModel = MOMENTPipeline.from_pretrained(
        #     "AutonLab/MOMENT-1-large", 
        #     model_kwargs={
        #         "task_name": "classification",
        #         "n_channels": self.configs.input_channels,
        #         "num_class": self.configs.num_classes,
        #         "sequence_len": self.configs.sequence_len,
        #         "num_layer": self.configs.prompt_length,
        #         "prompt_init": "uniform",
        #         "dropout": configs.dropout_src,
        #     },
        # )
        # self.sourceModel.init()

        # self.modelmoment = MOMENTPipeline.from_pretrained(
        #     "AutonLab/MOMENT-1-large", 
        #     model_kwargs={
        #         "task_name": "classification",
        #         "n_channels": self.configs.input_channels,
        #         "num_class": self.configs.num_classes,
        #         "sequence_len": self.configs.sequence_len,
        #         "num_layer": self.configs.prompt_length,
        #         "prompt_init": "uniform",
        #         "dropout": configs.dropout,
        #     },
        # )
        # self.modelmoment.init()

        # self.PreTrainedModel = MOMENTPipeline.from_pretrained(
        #     "AutonLab/MOMENT-1-large", 
        #     model_kwargs={
        #         "task_name": "classification",
        #         "n_channels": self.configs.input_channels,
        #         "num_class": self.configs.num_classes,
        #         "sequence_len": self.configs.sequence_len,
        #         "num_layer": self.configs.prompt_length,
        #         "prompt_init": "uniform",
        #         "dropout": configs.dropout,
        #         "freeze_head": True,
        #     },
        # )
        # self.PreTrainedModel.init()

        # self.modelmoment2 = MOMENTPipeline.from_pretrained(
        #     "AutonLab/MOMENT-1-large", 
        #     model_kwargs={
        #         "task_name": "classification",
        #         "n_channels": self.configs.input_channels,
        #         "num_class": self.configs.num_classes,
        #         "sequence_len": self.configs.sequence_len,
        #         "num_layer": self.configs.prompt_length,
        #         "prompt_init": "uniform",
        #         "dropout": configs.dropout,
        #     },
        # )
        # self.modelmoment2.init()

        # print(f'self.sourceModel:{self.sourceModel}')

        # self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # self.sourceModel = nn.DataParallel(self.sourceModel)
        # self.sourceModel = self.sourceModel.module
        # self.modelmoment = nn.DataParallel(self.modelmoment)
        # self.modelmoment = self.modelmoment.module
        # self.PreTrainedModel = nn.DataParallel(self.PreTrainedModel)
        # self.PreTrainedModel = self.PreTrainedModel.module
        # self.modelmoment2 = nn.DataParallel(self.modelmoment2)
        # self.modelmoment2 = self.modelmoment2.module
        # self.AE_cls = nn.DataParallel(self.AE_cls)
        # self.AE_cls = self.AE_cls.module
        # self.AE_cls2 = nn.DataParallel(self.AE_cls2)
        # self.AE_cls2 = self.AE_cls2.module


        self.optimizer = torch.optim.Adam(
            self.model_trg.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # self.optimizer = torch.optim.Adam(
        #     self.modelmoment.parameters(),
        #     lr=hparams["learning_rate"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.optimizerRefine = torch.optim.Adam(
        #     self.modelmoment.parameters(),
        #     lr=hparams["learning_rate_refine"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.optimizer2 = torch.optim.Adam(
        #     self.modelmoment2.parameters(),
        #     lr=hparams["learning_rate"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.optimizerAE = torch.optim.Adam(
        #     self.AE_cls.parameters(),
        #     lr=hparams["learning_rate_AE"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # self.optimizerAE2 = torch.optim.Adam(
        #     self.AE_cls2.parameters(),
        #     lr=hparams["learning_rate_AE"],
        #     weight_decay=hparams["weight_decay"]
        # )

        self.pre_optimizer = torch.optim.Adam(
            self.model_src.parameters(),
            # self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        # self.pre_optimizer = torch.optim.Adam(
        #     self.sourceModel.parameters(),
        #     # self.network.parameters(),
        #     lr=hparams["pre_learning_rate"],
        #     weight_decay=hparams["weight_decay"]
        # )

        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerRefine = StepLR(self.optimizerRefine, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_scheduler2 = StepLR(self.optimizer2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerAE = StepLR(self.optimizerAE, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # self.lr_schedulerAE2 = StepLR(self.optimizerAE2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _, _, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                # print(f'src_x:{src_x}')

                # forward pass correct sequences
                # outputs = self.sourceModel(src_x)
                # src_feat, _ = self.feature_extractor(src_x)


                for name, param in self.model_src.named_parameters():
                    if param.requires_grad:
                        # print(name, param.data)
                        print(name)

                # classifier predictions
                src_pred = self.model_src(src_x)
                # src_pred = outputs.logits
                # src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss 
                total_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.model_src.state_dict())
        return src_only_model

    # def update(self, trg_dataloader, avg_meter, logger, source_model_dir):
    # def update(self, trg_dataloader, avg_meter, logger):
    def update(self, trg_dataloader, trg_test_dataloader, avg_meter, logger, num_neighbors):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = self.model_trg.state_dict()
        self.last_model = self.model_trg.state_dict()
        # self.source_model_dir = source_model_dir

        # for k, v in self.classifier.named_parameters():
        # freeze both classifier and ood detector
        # for k, v in self.classifier.named_parameters():
        #     v.requires_grad = False
        # for k, v in self.temporal_verifier.named_parameters():
        #     v.requires_grad = False

        # print(f'self.pretrained_source:{self.pretrained_source}')

        # if not self.pretrained_source:
        #     print(f'Load pretrained source model..')
        #     load_source_model_path = source_model_dir + "/checkpoint.pt"
        #     print(f'source model path:{load_source_model_path}')
        #     self.sourceModel.load_state_dict(torch.load(load_source_model_path)["non_adapted"])
        #     for param in self.sourceModel.parameters():
        #         if not param.data.is_cuda:
        #             # print(f'model_t_all[0] param.data:{param.data}')
        #             # print(f'param.data.davice:{param.data.device}')
        #             param.data = param.to('cuda')

        total_epochs = self.hparams["num_epochs"] 
        # forget_rate = np.ones(total_epochs) * self.hparams["forget_rate"]
        # forget_rate[:(self.hparams["warm_target"]+self.hparams["num_gradual"])] = np.linspace(0, self.hparams["forget_rate"], (self.hparams["warm_target"]+self.hparams["num_gradual"]))
        # rate_schedule = np.ones(total_epochs) * self.hparams["forget_rate"]
        # rate_schedule[:(self.hparams["num_gradual"])] = np.linspace(0, self.hparams["forget_rate"]**self.hparams["exponent"], self.hparams["num_gradual"])
        # print(f'rate_schedule:{rate_schedule}')
        # print(f'rate_schedule.shape:{rate_schedule.shape}')
        # banks = eval_and_label_dataset(0, self.feature_extractor, self.classifier_t, None, trg_test_dataloader, trg_dataloader, num_neighbors)
        
        # banks, acc = eval_and_label_dataset(0, self.sourceModel, self.modelmoment, None, trg_test_dataloader, num_neighbors)
        # print(f'acc:{acc}')

        # train
        for epoch in range(0, total_epochs):
        # for epoch in range(1, self.hparams["num_epochs"] + 1):
            # if epoch <= round(total_epochs/2):#total_epochs
            #     alpha = 0.0
            # else:
            #     alpha = (epoch*2-total_epochs)/total_epochs

            
            self.model_trg.train()
            # self.modelmoment2.train()

            for step, (trg_x, _, trg_idx, trg_x_weak, _) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)

                # Validation data
                # val_x, val_y, _ = next(iter(val_dataloader))
                # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    trg_pred_src = self.model_src(trg_x)
                    # outputs_src = self.sourceModel(trg_x_weak)
                    # trg_pred_src = outputs_src.logits
                    # src_feat = outputs_src.embeddings
                    # trg_feat_src = outputs_src.embeddings2

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)

                
                # # Pseudo-label refinement
                # with torch.no_grad():
                #     # probs = F.softmax(trg_pred, dim=1)
                #     probs = src_prob
                #     # probs = probs_out_std
                #     # probs2 = F.softmax(out_un, dim=1)
                #     pseudo_labels, probs_refine, _, _ = refine_predictions(trg_feat_src, probs, banks, num_neighbors)
                #     # print(f'pseudo_labels:{pseudo_labels}')
                #     # print(f'pseudo_labels.shape:{pseudo_labels.shape}')
                

                # Co-teaching model output
                # with torch.no_grad():
                #     outputs_ind = self.modelmoment2(trg_x)
                #     trg_pred_ind = outputs_ind.logits
                #     trg_ind_prob = nn.Softmax(dim=1)(trg_pred_ind)

                # Target Model1 output
                trg_pred = self.model_trg(trg_x)
                # outputs = self.modelmoment(trg_x_weak)
                # trg_pred = outputs.logits
                # trg_feat = outputs.embeddings
                # trg_feat = outputs.embeddings2
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # # Correlation Matrix ------------------------------------------------------------
                # # PreTrainedModel embeddings
                # with torch.no_grad():
                #     outputs_ptm = self.PreTrainedModel(trg_x)
                #     ptm_feat = outputs_ptm.embeddings

                # # print(f'ptm_feat.shape:{ptm_feat.shape}')
                # # print(f'trg_feat.shape:{trg_feat.shape}')
                # # ptm_feat = ptm_feat.mean(dim=1)
                # ptm_feat = torch.flatten(ptm_feat, start_dim=1)
                # ptm_feat = F.normalize(ptm_feat, dim=-1)
                # # print(f'after mean transpose ptm_feat.shape:{ptm_feat.shape}')
                # # print(f'ptm_feat:{ptm_feat}')
                # # print(f'ptm_feat.shape:{ptm_feat.shape}')
                # # ptm_feat = ptm_feat/F.normalize(ptm_feat, dim=-1, p=2)
                # # print(f'after l2norm ptm_feat.shape:{ptm_feat.shape}')
                # # print(f'ptm_feat:{ptm_feat}')
                # # ptm_feat = ptm_feat.mean(dim=1)

                # # trg_feat = trg_feat.mean(dim=1)
                # trg_feat = torch.flatten(trg_feat, start_dim=1)
                # trg_feat = F.normalize(trg_feat, dim=-1)
                # # print(f'trg_feat:{trg_feat}')
                # print(f'trg_feat.shape:{trg_feat.shape}')
                # # trg_feat = trg_feat/F.normalize(trg_feat, dim=-1, p=2)
                # # trg_feat = trg_feat.mean(dim=1)
                # # print(f'ptm_feat:{ptm_feat}')
                # # print(f'F.normalize(ptm_feat, dim=1):{F.normalize(ptm_feat, dim=1)}')
                # # print(f'trg_feat:{trg_feat}')
                # # print(f'F.normalize(trg_feat, dim=1):{F.normalize(trg_feat, dim=1)}')

                # # m = torch.matmul(F.normalize(ptm_feat, dim=1).T, F.normalize(trg_feat, dim=1))
                # # m = torch.matmul(F.normalize(ptm_feat, dim=-1, p=2).T, F.normalize(trg_feat, dim=-1, p=2))
                # m = (F.normalize(ptm_feat, dim=-1, p=2).T @ F.normalize(trg_feat, dim=-1, p=2))/self.hparams["batch_size"]
                # # m = torch.einsum('bnd,bmd->nd', ptm_feat, trg_feat) / ptm_feat.shape[1]  # Shape [d_model, d_model]
                # # m = torch.einsum('bnd,bne->de', ptm_feat, trg_feat) / (ptm_feat.shape[0] * ptm_feat.shape[1]) # Shape [d_model, d_model]
                # # m = torch.mul(F.normalize(ptm_feat, dim=-1, p=2).T, F.normalize(trg_feat, dim=-1, p=2))

                # # print(f'ptm_feat.shape[1]:{ptm_feat.shape[1]}')
                # # print(f'm:{m}')
                # print(f'm.shape:{m.shape}')
                # # print(f'len(m):{len(m)}')
                # # print(f'after mean ptm_feat.shape:{ptm_feat.shape}')

                # # loss_diag = (torch.square(1-m)).mean(dim=0)
                # # print(f'torch.diagonal(m,0):{torch.diagonal(m,0)}')
                # # print(f'1-torch.diagonal(m, 0):{1-torch.diagonal(m, 0)}')

                # # Loss Relationship PreTrainedModel and Target Model
                # # loss_diag = torch.square(1-torch.diagonal(m, 0)).mean(dim=0)
                # loss_diag = torch.mean((1 - torch.diagonal(m, 0)) ** 2)
                # # print(f'1-torch.diagonal(m, 0):{1-torch.diagonal(m, 0)}')
                # # print(f'torch.square(1-torch.diagonal(m, 0)):{torch.square(1-torch.diagonal(m, 0))}')
                # # print(f'loss_diag:{loss_diag}')

                # m2 = m * (1 - torch.eye(len(m))).to(self.device)
                # # print(f'm2:{m2}')

                # # Loss redundancy PreTrainedModel and Target Model
                # loss_rdn = torch.square(m2).mean()
                # # print(f'loss_rdn:{loss_rdn}')
                # # end correlation ------------------------------------------------------------


                # # Pseudo-label refinement --------------------------------------------------
                # with torch.no_grad():
                #     # probs = F.softmax(trg_pred, dim=1)
                #     probs = src_prob
                #     # probs = probs_out_std
                #     # probs2 = F.softmax(out_un, dim=1)
                #     pseudo_labels, probs_refine, _, _ = refine_predictions(trg_feat, probs, banks, num_neighbors)
                #     # print(f'pseudo_labels:{pseudo_labels}')
                #     # print(f'pseudo_labels.shape:{pseudo_labels.shape}')
                # # ---------------------------------------------------------------------------

                # Target Model2 output
                # outputs2 = self.modelmoment2(trg_x)
                # trg_pred2 = outputs2.logits

                # co-teaching loss
                # loss_1, loss_2 = self.loss_coteaching(trg_pred, trg_pred2, pseudo_labels, rate_schedule[epoch], trg_idx)

                # # reconstruct -------------------------------------------------------------
                # mt = random.uniform(0,1) #mask
                # s0,s1,s2 = trg_x.shape
                # randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                # mt = torch.bernoulli(randuniform).to(self.device)
                # m_ones = torch.ones(s0,s1,s2).to(self.device)

                # sum_mt = mt.flatten().sum()
                # # c_mask = 1/((s0*s1*s2) * sum_mt)
                # # c_unmask = 1/(((s0*s1*s2) - sum_mt)*(s0*s1*s2))
                # c_mask = sum_mt/(s0*s1*s2)
                # c_unmask = ((s0*s1*s2) - sum_mt)/(s0*s1*s2)
                # # print(f'c_mask:{c_mask}')
                # # print(f'c_unmask:{c_unmask}')
                # # print(f'mt:{mt}')
                # # print(f'trg_x.shape:{trg_x.shape}')

                # #src if mt=1 -> x=0
                # src2 = torch.clone(trg_x)
                # src2 = src2 * (m_ones-mt)
                # gamma = 0.5
                # criterion = RMSELoss()
                # # print(f'src2.shape:{src2.shape}')

                # # pred reconstruct
                # out = self.modelmoment.forward_reconstruct(src2)
                # pred_reconstruct = out.reconstruction

                # # out2 = self.modelmoment2.forward_reconstruct(src2)
                # # pred_reconstruct2 = out2.reconstruction
                # # print(f'pred_reconstruct.shape:{pred_reconstruct.shape}')
                # # print(f'src2.shape:{src2.shape}')
                # # print(f'criterion(pred_reconstruct, src2):{criterion(pred_reconstruct, src2)}')
                # #loss reconstruct
                # loss_reconstruct = gamma * c_mask * criterion(pred_reconstruct, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct, src2)
                # # loss_reconstruct2 = gamma * c_mask * criterion(pred_reconstruct2, src2) + (1-gamma) * c_unmask * criterion(pred_reconstruct2, src2)
                # # print(f'loss_reconstruct:{loss_reconstruct}')
                # # -------------------------------------------------------------------------

                
                # # Lama -------------------------------------------------------------------------
                # loss_cot = F.cross_entropy(trg_pred_ind, pseudo_labels, reduction='none') # reduction='none' to get loss per batch instead of avg loss
                # # loss_cot = F.cross_entropy(output_target, pseudo_labels, reduction='none')
                # # print(f'loss_cot:{loss_cot}')
                
                # # get (1-R) percent low loss samples
                # ind_loss_sorted = np.argsort(loss_cot.cpu().data).to(self.device)
                # # ind_loss_sorted_thres = np.argsort(trg_ind_prob.cpu().data).to(self.device)

                # remember_rate = 1 - (1-alpha) * forget_rate
                # num_remember = math.ceil(remember_rate * len(ind_loss_sorted))
                # ind_loss_update = ind_loss_sorted[:num_remember]
                # # ind_loss_neg_update = ind_loss_sorted[(num_remember):]#/*2+num_neg

                # # print(f'remember_rate:{remember_rate}')
                # # print(f'len(ind_loss_update):{len(ind_loss_update)}')
                # # print(f'num data:{num_remember}')
                # # print(f'trg_x[ind_loss_update]:{trg_x[ind_loss_update]}')
                # # outputs_sorted = modelmoment(trg_x[ind_loss_update])
                # # trg_pred2 = outputs_sorted.logits

                # # outputs_ind_sorted = modelmoment_ind(trg_x[ind_loss_update])
                # # trg_pred_ind2 = outputs_ind_sorted.logits

                # # max_output2, _ = torch.max(trg_pred2, 1)
                # # max_output_ind2, _ = torch.max(trg_pred_ind2, 1)

                # # a2 = max_output2/(max_output2+max_output_ind2)
                # # b2 = max_output_ind2/(max_output2+max_output_ind2)
                # # output_target2 = torch.unsqueeze(a2,1)*trg_pred2 + torch.unsqueeze(b2,1)*trg_pred_ind2

                
                # # trg_prob_sorted = torch.log_softmax(outputs_sorted.logits, dim=1)

                # # with torch.no_grad():
                # #     outputs_src_sorted = sourceModel(trg_x[ind_loss_update])
                # #     trg_pred_src_sorted = outputs_src_sorted.logits

                # # trg_prob_src_sorted = nn.Softmax(dim=1)(trg_pred_src_sorted)
                # # _, pseudo_labels_sorted = torch.max(trg_prob_src_sorted, 1)

                # pseudo_labels_sorted = pseudo_labels[ind_loss_update].to(self.device)

                # # with torch.no_grad():
                # #     outputs_src2 = sourceModel(trg_x[ind_loss_update])
                # #     trg_pred_src2 = outputs_src2.logits

                # # src_prob2 = nn.Softmax(dim=1)(trg_pred_src2)
                # # _, pseudo_labels2 = torch.max(src_prob2, 1)
                # # print(f'pseudo_labels_sorted:{pseudo_labels_sorted}')
                # # print(f'pseudo_labels2:{pseudo_labels2}')
                # # print(f'trg_prob_sorted:{trg_prob_sorted}')
                # # print(f'prompt.shape:{prompt.shape}')

                # # Entropy loss
                # # trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
                # # print(f'torch.mean(EntropyLoss(trg_prob_sorted):{torch.mean(EntropyLoss(trg_prob_sorted))}')
                # # trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob_sorted))

                # # trg_prob_sorted = torch.log_softmax(trg_pred[ind_loss_update], dim=1).cuda()
                # # print(f'trg_prob_sorted:{trg_prob_sorted}')
                # # print(f'trg_prob_sorted.shape:{trg_prob_sorted.shape}')
                

                # # CE_loss = self.cross_entropy(outputs_sorted.logits, pseudo_labels_sorted)
                # # CE_loss = self.cross_entropy(output_target[ind_loss_update], pseudo_labels_sorted)
                CE_loss = self.cross_entropy(trg_pred, pseudo_labels)
                # # print(f'CE_loss:{CE_loss}')
                # # end Lama -----------------------------------------------------------------------------------

                # # prompt reconstruction -------------------------------------------------------------------------
                # AE_loss = 0
                # # AE_loss2 = 0
                # # use prompt
                # prompt = self.modelmoment.prompt.prompt
                # prompt_new = self.AE_cls(prompt)
                # # prompt2 = self.modelmoment2.prompt.prompt
                # # prompt_new2 = self.AE_cls2(prompt2)

                # AE_loss = torch.pow(torch.linalg.norm(prompt_new - prompt)/prompt.shape[0], 2)
                # # AE_loss2 = torch.pow(torch.linalg.norm(prompt_new2 - prompt2)/prompt2.shape[0], 2)
                # # print(f'AE_loss:{AE_loss}')

                # while AE_loss > 5:
                #     AE_loss = AE_loss / 10

                # # while AE_loss2 > 5:
                # #     AE_loss2 = AE_loss2 / 10
                
                # # -----------------------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                # loss = CE_loss + AE_loss + loss_reconstruct
                # loss = CE_loss + loss_diag + loss_rdn
                # loss = CE_loss + AE_loss + loss_diag + loss_rdn
                # loss = CE_loss + AE_loss + loss_reconstruct + loss_diag + loss_rdn
                loss = CE_loss
                
                # loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                # loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')

                # update_labels(banks, trg_idx, trg_feat, trg_pred_src)
                

                self.optimizer.zero_grad()
                # self.optimizerAE.zero_grad()
                loss.backward()
                self.optimizer.step()

                # self.optimizer2.zero_grad()
                # loss2.backward()
                # self.optimizer2.step()

                
                # self.optimizerAE.step()
                # self.optimizerAE2.step()
                # self.decoder_optimizer.step()
                # self.global_step+=1

                losses = {'entropy_loss': CE_loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_diag': loss_diag.detach().item(), 'Loss_rdn': loss_rdn.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_diag': loss_diag.detach().item(), 'Loss_rdn': loss_rdn.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Loss_diag': loss_diag.detach().item(), 'Loss_rdn': loss_rdn.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                #  'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)


            # Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            

            self.lr_scheduler.step()
            # self.lr_scheduler2.step()
            # self.lr_schedulerAE.step()
            # self.lr_schedulerAE2.step()

            # # saving the best model based on src risk
            # if (epoch + 1) % 10 == 0 and (loss_1.avg < best_src_risk or loss_2.avg < best_src_risk):
            #     # best_src_risk = avg_meter['Total_loss'].avg

            #     if loss_1.avg < loss_2.avg:
            #         best_src_risk = loss_1.avg
            #         self.best_model = deepcopy(self.modelmoment.state_dict())
            #         self.best_model_net1 = True
            #     else:
            #         best_src_risk = loss_2.avg
            #         self.best_model = deepcopy(self.modelmoment2.state_dict())
            #         self.best_model_net1 = False


            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def refine(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        self.best_model = self.modelmoment.state_dict()
        self.last_model = self.modelmoment.state_dict()
        # self.source_model_dir = source_model_dir

        total_epochs = self.hparams["num_iter"] 
        # w = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)
        # print(f'w.shape:{w.shape}')

        # train
        for epoch in range(0, total_epochs):
        # for epoch in range(1, self.hparams["num_epochs"] + 1):

            # non_candidate_loss = 0
            self.modelmoment.train()

            for step, (trg_x, _, trg_idx, trg_x_weak, trg_x_strong) in enumerate(trg_dataloader):

                # Target data
                trg_x = trg_x.float().to(self.device)
                trg_x_weak = trg_x_weak.float().to(self.device)
                trg_x_strong = trg_x_strong.float().to(self.device)

                # Validation data
                # val_x, val_y, _ = next(iter(val_dataloader))
                # val_x, val_y = val_x.float().to(self.device), val_y.long().to(self.device)

                # Source Model soft-labels and pseudo-labels
                with torch.no_grad():
                    outputs_src = self.sourceModel(trg_x)
                    trg_pred_src = outputs_src.logits

                    src_prob = nn.Softmax(dim=1)(trg_pred_src)
                    src_conf, pseudo_labels = torch.max(src_prob, 1)
                    # pseudo_labels2 = np.argmax(src_prob.cpu().data, axis=1)


                # Target Model output
                outputs = self.modelmoment(trg_x)
                trg_pred = outputs.logits

                trg_prob = nn.Softmax(dim=1)(trg_pred)
                trg_conf, _ = torch.max(trg_prob, 1)
                # print(f'trg_prob:{trg_prob}')


                # Target Model output with strong aug
                outputs_strong = self.modelmoment(trg_x_strong)
                trg_pred_strong = outputs_strong.logits

                trg_prob_strong = nn.Softmax(dim=1)(trg_pred_strong)
                trg_conf_strong, _ = torch.max(trg_prob_strong, 1)

                # Target Model output with weak aug
                outputs_weak = self.modelmoment(trg_x_weak)
                trg_pred_weak = outputs_weak.logits

                trg_prob_weak = nn.Softmax(dim=1)(trg_pred_weak)
                trg_conf_weak, _ = torch.max(trg_prob_weak, 1)
                # print(f'trg_prob_weak:{trg_prob_weak}')

                # print(f'trg_x_strong.shape:{trg_x_strong.shape}')
                # print(f'trg_x_strong:{trg_x_strong}')
                # print(f'trg_pred_strong:{trg_pred_strong}')
                # print(f'trg_prob_strong:{trg_prob_strong}')


                # select pseudo-candidate set (Z) based on threshold -----------------------------------------------
                prob_sel = trg_prob > self.hparams["tau"] 
                non_prob_sel = ~prob_sel              
                # calculate norm |Z|
                num_candidate = torch.sum(prob_sel, dim=1)
                # print(f'num_candidate:{num_candidate}')

                # select samples based on threshold
                conf_sel = trg_conf > self.hparams["tau"]
                non_conf_sel = ~conf_sel


                # print(f'prob_sel:{prob_sel}')
                # print(f'non_prob_sel:{non_prob_sel}')
                # print(f'conf_sel:{conf_sel}')
                # print(f'len(conf_sel):{len(conf_sel)}')
                # print(f'range(len(conf_sel)):{range(len(conf_sel))}')
                # print(f'~conf_sel:{~conf_sel}')
                # print(f'trg_conf.shape:{trg_conf.shape}')

                ind_candidate = torch.argwhere(prob_sel)
                ind_non_candidate = torch.argwhere(non_prob_sel)

                ind_cand_sampl = torch.argwhere(conf_sel)
                ind_non_cand_sampl = torch.argwhere(non_conf_sel)
                
                # print(f'ind_candidate:{ind_candidate}')
                # print(f'ind_cand_sampl:{ind_cand_sampl}')
                # print(f'ind_non_candidate:{ind_non_candidate}')

                w = torch.zeros(trg_prob.shape).to(self.device)
                # print(f'w:{w}')
                probs_strong = torch.zeros(self.hparams["batch_size"], self.configs.num_classes).to(self.device)
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'trg_prob_strong[0]:{trg_prob_strong[0]}')

                # initialized w
                if epoch == 0:
                    for i in ind_candidate:
                        w[i[0],i[1]] = 1/num_candidate[i[0]]
                        # print(f'i:{i}')
                        # print(f'num_candidate[i]:{num_candidate[i]}')
                else:
                    for i in ind_candidate:
                        probs_strong[i[0],i[1]] = trg_prob_strong[i[0],i[1]]

                    for i in ind_candidate:
                        w[i[0],i[1]] = trg_prob_strong[i[0],i[1]]/torch.sum(trg_prob_strong[i[0]])

                # candidate_sel = trg_conf_strong[ind_cand_sampl] < self.hparams["threshold"]
                # print(f'after intialized w:{w}')

                candidate_sel = trg_conf_strong < self.hparams["threshold"]

                ind_candidate_sel = torch.logical_and(conf_sel, candidate_sel)
                ind_candidate_loss = torch.squeeze(ind_candidate_sel.nonzero(), dim=-1)
                # print(f'ind_candidate_loss2:{ind_candidate_loss2}')
                # print(f'ind_keep:{ind_keep}')

                # print(f'trg_conf_strong:{trg_conf_strong}')
                # print(f'trg_conf:{trg_conf}')
                # print(f'trg_conf_strong[ind_cand_sampl]:{trg_conf_strong[ind_cand_sampl]}')
                # print(f'trg_conf_strong[ind_candidate]:{trg_conf_strong[ind_candidate]}')
                # candidate_sel = trg_conf_strong[ind_candidate] < self.hparams["threshold"]
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'trg_prob_strong[ind_candidate]:{trg_prob_strong[ind_candidate]}')
                # print(f'trg_conf_strong:{trg_conf_strong}')
                # print(f'candidate_sel:{candidate_sel}')
                # if epoch > 0:
                # print(f'trg_x_strong:{trg_x_strong}')
                # print(f'ind_candidate:{ind_candidate}')
                # print(f'trg_prob_strong:{trg_prob_strong}')
                # print(f'probs_strong:{probs_strong}')
                # print(f'w:{w}')

                # ind_candidate_loss = torch.argwhere(candidate_sel)
                # ind_candidate_loss = torch.argwhere(torch.squeeze(candidate_sel, dim=-1))
                # print(f'ind_candidate_loss:{ind_candidate_loss}')
                # print(f'w[ind_candidate_loss]:{w[ind_candidate_loss]}')
                # print(f'trg_prob_strong[ind_candidate_loss]:{trg_prob_strong[ind_candidate_loss]}')

                # print(f'w[ind_candidate_loss]:{w[ind_candidate_loss]}')
                # print(f'trg_prob_strong[ind_candidate_loss].shape:{trg_prob_strong[ind_candidate_loss].shape}')

                # log_w = np.log(w.cpu())
                # print(f'w[torch.squeeze(ind_candidate_loss, dim=-1)]:{w[torch.squeeze(ind_candidate_loss, dim=-1)]}')
                # print(f'w:{w}')
                # print(f'w[ind_candidate_loss].shape:{w[ind_candidate_loss].shape}')
                # print(f'F.log_softmax(w[ind_candidate_loss]):{F.log_softmax(w[ind_candidate_loss], dim=1)}')
                # print(f'nn.Softmax(w[ind_candidate_loss]):{nn.Softmax(dim=-1)(w[ind_candidate_loss])}')
                if ind_candidate_loss.numel():
                    # print(f'ind_candidate_loss:{ind_candidate_loss}')
                    # print(f'w[ind_candidate_loss].shape:{w[ind_candidate_loss].shape}')
                    # print(f'trg_prob_strong[ind_candidate_loss].shape:{trg_prob_strong[ind_candidate_loss].shape}')
                    # print(f'w[torch.squeeze(ind_candidate_loss, dim=-1)].shape:{w[torch.squeeze(ind_candidate_loss, dim=-1)].shape}')
                    # print(f'trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)].shape:{trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)].shape}')
                    # candidate_loss = self.kl_loss(F.log_softmax(w[torch.squeeze(ind_candidate_loss, dim=-1)], dim=1), trg_prob_strong[torch.squeeze(ind_candidate_loss, dim=-1)])
                    candidate_loss = self.kl_loss(F.log_softmax(w[ind_candidate_loss], dim=1), trg_prob_strong[ind_candidate_loss])
                else:
                    candidate_loss = torch.zeros(1).to(self.device)
                    # candidate_loss = 0.0
                # candidate_loss = self.kl_loss(F.log_softmax(w, dim=1), trg_prob_strong)
                # candidate_loss = self.kl_loss(trg_prob_strong[ind_candidate_loss], F.log_softmax(w[ind_candidate_loss], -1))
                # candidate_loss = self.kl_loss(nn.Softmax(dim=-1)(w[ind_candidate_loss]), trg_prob_strong[ind_candidate_loss])
                # candidate_loss = self.kl_loss(trg_prob_strong[ind_candidate_loss], w[ind_candidate_loss])
                # print(f'candidate_loss:{candidate_loss}')
                # print(f'ind_non_candidate:{ind_non_candidate}')

                # print(f'trg_prob_weak[ind_non_candidate]:{trg_prob_weak[ind_non_candidate]}')

                # for i in ind_non_candidate:
                #     # print(f'i[0]:{i[0]}')
                #     # print(f'i[1]:{i[1]}')
                #     print(f'trg_prob_weak[i[0],i[1]]:{trg_prob_weak[i[0],i[1]]}')
                # print(f'trg_prob_weak:{trg_prob_weak}')
                # print(f'trg_pred_weak:{trg_pred_weak}')
                # non_candidate_loss = 0
                non_candidate_loss = torch.zeros(1).to(self.device)
                for i in ind_non_candidate:
                    non_candidate_loss += torch.log(1 - trg_prob_weak[i[0],i[1]]).to(self.device)
                    # non_candidate_loss += torch.log(1 - trg_pred_weak[i[0],i[1]])
                    # print(f'trg_pred_weak[{i[0]},{i[1]}]:{trg_pred_weak[i[0],i[1]]}')
                    # print(f'torch.log(1 - trg_prob_weak[{i[0]},{i[1]}]):{torch.log(1 - trg_prob_weak[i[0],i[1]])}')
                    # print(f'torch.log(1 - trg_pred_weak[{i[0]},{i[1]}]):{torch.log(1 - trg_pred_weak[i[0],i[1]])}')
                    # print(f'non_candidate_loss:{non_candidate_loss}')

                non_candidate_loss = -non_candidate_loss/self.hparams["batch_size"]
                # print(f'average -non_candidate_loss:{non_candidate_loss}')
                # w = torch.div(torch.t(conf_sel),num_candidate)
                # w = torch.t(w)
                # w[ind_candidate] = 1/num_candidate
                # print(f'after initialize w:{w}')
                # print(f'len(trg_x):{len(trg_x)}')
                # pseudo_candidate_set = trg_x[ind_candidate]
                # pseudo_labels = pseudo_labels[ind_candidate]
                # print(f'after ind_keep len(trg_x):{len(trg_x)}')
                # ind_keep = torch.squeeze(conf_sel.nonzero(), dim=-1)
                # print(f'conf_sel:{conf_sel}')
                # print(f'ind_keep:{ind_keep}')
                # end select pseudo-labels based on threshold ------------------------------------------
                

                # # CE_loss = self.cross_entropy(outputs_sorted.logits, pseudo_labels_sorted)
                # # CE_loss = self.cross_entropy(output_target[ind_loss_update], pseudo_labels_sorted)
                # CE_loss = self.cross_entropy(trg_pred, pseudo_labels)
                # # print(f'CE_loss:{CE_loss}')
                # # end Lama -----------------------------------------------------------------------------------


                '''
                Overall objective loss
                '''
                # removing trg ent
                # loss = trg_ent + AE_loss #+ self.hparams['TOV_wt'] * tov_loss
                # loss = CE_loss + AE_loss
                # loss_refine = candidate_loss + self.hparams["lam"] * (non_candidate_loss)
                loss_refine = candidate_loss + self.hparams["lam"] * (non_candidate_loss)
                # loss2 = loss_2 + AE_loss2 + loss_reconstruct2
                # loss = CE_loss
                # loss = CE_loss + loss_reconstruct
                # print(f'loss_1:{loss_1}')
                # print(f'AE_loss:{AE_loss}')
                # print(f'loss_reconstruct:{loss_reconstruct}')
                # print('====================================')
                

                # self.optimizer.zero_grad()
                # loss_refine.backward()
                # self.optimizer.step()
                self.optimizerRefine.zero_grad()
                loss_refine.backward()
                self.optimizerRefine.step()

                
                # losses = {'entropy_loss': CE_loss.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Total_loss': loss.detach().item()}
                # losses = {'entropy_loss': CE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item()}
                losses_refine = {'candidate_loss': candidate_loss.detach().item(), 'non_candidate_loss': non_candidate_loss.detach().item(), 'Total_loss': loss_refine.detach().item()}
                # losses = {'entropy_loss1': loss_1.detach().item(), 'AE_loss': AE_loss.detach().item(), 'Loss_reconstruct': loss_reconstruct.detach().item(), 'Total_loss': loss.detach().item(),
                #  'entropy_loss2': loss_2.detach().item(), 'AE_loss2': AE_loss2.detach().item(), 'Loss_reconstruct2': loss_reconstruct2.detach().item(), 'Total_loss2': loss2.detach().item()}
                for key, val in losses_refine.items():
                    avg_meter[key].update(val, 32)


            # Update w



            # Total_loss_network = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment, self.optimizer, self.modelmoment2, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # Total_loss_network2 = self.target_train(trg_dataloader, val_dataloader, avg_meter, self.modelmoment2, self.optimizer2, self.modelmoment, self.sourceModel, forget_rate[epoch-1], alpha, epoch, logger)

            # self.lr_scheduler.step()
            self.lr_schedulerRefine.step()



            logger.debug(f'[Epoch : {epoch+1}/{self.hparams["num_iter"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return self.last_model, self.best_model

    def get_timeclr(self, encoder):
        # aug_bank_ver = int(model_config['timeclr']['aug_bank_ver'])
        # print('get_timeclr')
        aug_bank_ver = 0
        if aug_bank_ver == 0:
            aug_bank = [
                lambda x:jittering(x, strength=0.1, seed=None),
                lambda x:smoothing(x, max_ratio=0.5, min_ratio=0.01, seed=None),
                lambda x:mag_warping(x, strength=1, seed=None),
                lambda x:add_slope(x, strength=1, seed=None),
                lambda x:add_spike(x, strength=3, seed=None),
                lambda x:add_step(x, min_ratio=0.1, strength=1, seed=None),
                lambda x:cropping(x, min_ratio=0.1, seed=None),
                lambda x:masking(x, max_ratio=0.5, seed=None),
                lambda x:shifting(x, seed=None),
                lambda x:time_warping(x, min_ratio=0.5, seed=None),
            ]

        encoder_ = TimeCLREncoder(encoder, aug_bank)
        # print(f'encoder_:{encoder_}')
        # print(f'aug_bank:{aug_bank}')
        pre_train_model_path = '/home/furqon/blackbox/foundation/mapu4/MAPU_SFDA_TS/models/trf_tc_0000_0399.npz'
        pkl = torch.load(pre_train_model_path, map_location='cpu')
        # print(f'self.encoder:{self.encoder}')
        encoder_.load_state_dict(pkl['model_state_dict'], strict=False)
        # encoder_ = load_pretrain(model_config['timeclr'], encoder_)
        return encoder_



