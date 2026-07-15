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





