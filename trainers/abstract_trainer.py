import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
# from torchmetrics import Accuracy, AUROC, F1Score
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, split_data
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.run_description = args.run_description if args.run_description is not None else args.da_method


        self.experiment_description = args.dataset


        # paths
        self.home_path = os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.save_source_dir = args.save_source_dir
        self.save_target_dir = args.save_target_dir
        


        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}
        self.num_neighbors = args.num_neighbors

        # metrics
        self.num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)        


    def sweep(self):
        # sweep configurations
        pass
    
    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        print(f'Lr:{self.hparams["pre_learning_rate"]}')
        print(f'Do:{self.dataset_configs.dropout_src}')
        self.logger.debug("=" * 45)

        self.non_adapted_model = self.algorithm.pretrain(self.src_train_dl, self.pre_loss_avg_meters, self.logger)
        self.calculate_metricsSource() 
        self.calculate_metricsSourcewithTargetData()
        # adapting step
        self.logger.debug("=" * 45)
        self.logger.debug(f'Adaptation stage..........')
        print(f'Lr:{self.hparams["learning_rate"]}')
        print(f'Do:{self.dataset_configs.dropout}')
        if self.da_method == "B2TSDA_NoCOT":
            print(f'Lr_AE:{self.hparams["learning_rate_AE"]}')
            print(f'Prompt_length:{self.dataset_configs.prompt_length}')

        self.logger.debug("=" * 45)

        self.last_model, self.best_model = self.algorithm.update(self.trg_train_dl, self.trg_test_dl, self.loss_avg_meters, self.logger, self.num_neighbors) # 5 Des
        self.calculate_metrics()
        self.logger.debug("=" * 45)

        # Refinement stage ---------------------------------------------------
        self.logger.debug(f'Refinement stage..........')
        print(f'Lr_refine:{self.hparams["learning_rate_refine"]}')
        print(f'Tau:{self.hparams["tau"]}')
        print(f'Gamma:{self.hparams["threshold"]}')
        print(f'Iteration:{self.hparams["num_iter"]}')
        print(f'Lambda:{self.hparams["lam"]}')
        print(f'JS:{self.dataset_configs.jitter_scale_ratio}')
        print(f'JR:{self.dataset_configs.jitter_ratio}')
        print(f'MS:{self.dataset_configs.max_seg}')
        self.logger.debug("=" * 45)
        self.last_model, self.best_model = self.algorithm.refine(self.trg_train_dl, self.loss_refine_avg_meters, self.logger)
        # --------------------------------------------------------------------

        return  self.non_adapted_model,  self.last_model, self.best_model

    def train_model_source(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        # backbone_fe = get_backbone_class(self.backbone)
        # print(f'algorithm_class:{algorithm_class}')
        # print(f'backbone_fe:{backbone_fe}')

        # Initilaize the algorithm
        self.algorithm = algorithm_class(self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug(f'Lr:{self.hparams["pre_learning_rate"]}')
        self.logger.debug(f'Do:{self.dataset_configs.dropout_src}')
        self.logger.debug(f'mid_dim:{self.dataset_configs.mid_dim}')
        self.logger.debug(f'out_dim:{self.dataset_configs.out_dim}')

        self.logger.debug("=" * 45)

        self.non_adapted_model = self.algorithm.pretrain(self.src_train_dl, self.src_test_dl, self.pre_loss_avg_meters, self.logger)
        # self.calculate_metricsSource() # 5 Des
        self.calculate_metricsSourcewithTargetData() # 5 Des
        # self.calculate_metricsSourceMAPU()
        # self.calculate_metricsSourcewithTargetDataMAPU()
        self.logger.debug("=" * 45)

        return  self.non_adapted_model

    def train_model_target(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)
        # print(f'algorithm_class:{algorithm_class}')
        # print(f'backbone_fe:{backbone_fe}')

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # adapting step
        self.logger.debug("=" * 45)
        self.logger.debug(f'Adaptation stage..........')
        self.logger.debug(f'Lr:{self.hparams["learning_rate"]}')
        self.logger.debug(f'Do:{self.dataset_configs.dropout}')
        self.logger.debug(f'mid_dim:{self.dataset_configs.mid_dim}')
        self.logger.debug(f'out_dim:{self.dataset_configs.out_dim}')
        if self.da_method == "B2TSDA_COT_target":
            print(f'Lr_AE:{self.hparams["learning_rate_AE"]}')
            # print(f'Lr_refine:{self.hparams["learning_rate_refine"]}')
            print(f'Prompt_length:{self.dataset_configs.prompt_length}')
            print(f'gma:{self.hparams["gma"]}')
            # print(f'JS:{self.dataset_configs.jitter_scale_ratio}')
            # print(f'JR:{self.dataset_configs.jitter_ratio}')
            # print(f'MS:{self.dataset_configs.max_seg}')
            # print(f'forget_rate:{self.hparams["forget_rate"]}')
        self.logger.debug("=" * 45)

        self.model1, self.model2 = self.algorithm.update(self.trg_train_dl, self.trg_test_dl, self.loss_avg_meters, self.logger, self.load_source_model_path) # 17 Maret COT

        self.logger.debug("=" * 45)

        return  self.model1, self.model2
    
    def train_model_source_multi(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        # backbone_fe = get_backbone_class(self.backbone)
        

        # Initilaize the algorithm
        self.algorithm = algorithm_class(self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug(f'Lr:{self.hparams["pre_learning_rate"]}')
        self.logger.debug(f'Do:{self.dataset_configs.dropout_src}')
        self.logger.debug(f'mid_dim:{self.dataset_configs.mid_dim}')
        self.logger.debug(f'out_dim:{self.dataset_configs.out_dim}')

        self.logger.debug("=" * 45)

        self.non_adapted_model = self.algorithm.pretrain(self.src_train_dl, self.src_test_dl, self.pre_loss_avg_meters, self.logger)
        
        self.logger.debug("=" * 45)

        return  self.non_adapted_model

    def train_model_target_multi(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # adapting step
        self.logger.debug("=" * 45)
        self.logger.debug(f'Adaptation stage..........')
        self.logger.debug(f'Lr:{self.hparams["learning_rate"]}')
        self.logger.debug(f'Do:{self.dataset_configs.dropout}')
        self.logger.debug(f'mid_dim:{self.dataset_configs.mid_dim}')
        self.logger.debug(f'out_dim:{self.dataset_configs.out_dim}')
        if self.da_method == "B2TSDA_COT_target":
            print(f'Lr_AE:{self.hparams["learning_rate_AE"]}')
            print(f'Prompt_length:{self.dataset_configs.prompt_length}')
            print(f'gma:{self.hparams["gma"]}')
            
        self.logger.debug("=" * 45)

        self.model1, self.model2 = self.algorithm.update(self.trg_train_dl, self.trg_test_dl, self.loss_avg_meters, self.logger, self.load_source_model_path) # 17 Maret COT


        self.logger.debug("=" * 45)

        return  self.model1, self.model2

    

    def train_model_SourceOnly(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # pretraining step
        self.logger.debug(f'Learning in Source Only Domain..........')
        print(f'Lr:{self.hparams["pre_learning_rate"]}')
        print(f'Do:{self.dataset_configs.dropout}')
        self.logger.debug("=" * 45)

        self.non_adapted_model = self.algorithm.pretrain(self.src_train_dl, self.pre_loss_avg_meters, self.logger)
        
        return  self.non_adapted_model

    def train_model_TargetOnly(self, load_source_model_path):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # adapting step
        self.logger.debug("=" * 45)
        self.logger.debug(f'Adaptation stage in Target Domain..........')
        print(f'Lr:{self.hparams["learning_rate"]}')
        print(f'Do:{self.dataset_configs.dropout}')
        print(f'Lr_AE:{self.hparams["learning_rate_AE"]}')
        print(f'Prompt_length:{self.dataset_configs.prompt_length}')
        self.logger.debug("=" * 45)

        self.last_model, self.best_model = self.algorithm.update(self.trg_train_dl, self.loss_avg_meters, self.logger, load_source_model_path)

        return  self.last_model, self.best_model

    def evaluate(self, test_loader):

        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features, seq_features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(np.array(total_loss)).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))


    def evaluate_ori(self, test_loader):

        model = self.algorithm.modelmoment
        
        model.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                outputs = model(data)
                predictions = outputs.logits

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(np.array(total_loss)).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def evaluate_aggr(self, test_loader):

        model = self.algorithm.modelmoment
        model2 = self.algorithm.modelmoment2
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

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def evaluateSource(self, test_loader):
        model = self.algorithm.sourceModel
        model.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _, _, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                outputs = model(data)
                predictions = outputs.logits

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id, run_id):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")

    def load_data_source(self, src_id, run_id):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")


    def load_data_target(self, trg_id: str, run_id: int):
        self.trg_train_dl = data_generator(
            self.data_path,
            trg_id,
            self.dataset_configs,
            self.hparams,
            "train"
        )
        self.trg_test_dl = data_generator(
            self.data_path,
            trg_id,
            self.dataset_configs,
            self.hparams,
            "test"
        )

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data 
        self.evaluate_ori(self.src_test_dl)
        src_risk = self.loss.item()
        self.evaluate_ori(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        

        risks = src_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, non_adapted, last_model, best_model):
        save_dict = {
            "non_adapted":non_adapted,
            "last": last_model,
            "best": best_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def save_checkpointTargetOnly(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def save_checkpointTargetCOT(self, home_path, log_dir, model1, model2):
        save_dict = {
            "model1": model1,
            "model2": model2
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def save_checkpointSourceOnly(self, home_path, log_dir, non_adapted):
        save_dict = {
            "non_adapted":non_adapted
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def calculate_avg_std_wandb_table(self, results):

        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):
       
        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}


        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks 
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']), allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self):
       
        self.evaluate_aggr(self.trg_test_dl)
        
        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        print(f'acc\t:{acc}')
        print(f'f1\t:{f1}')
        print(f'auroc\t:{auroc}')

        return acc, f1, auroc

    def calculate_metricsSourcewithTargetData(self):
       
        self.evaluateSource(self.trg_train_dl)
        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        print(f'------------------------')
        print(f'Pseudo-label evaluation:')
        print(f'acc\t:{acc}')
        print(f'f1\t:{f1}')
        print(f'auroc\t:{auroc}')
        print(f'------------------------')

        return acc, f1, auroc

    def calculate_metricsSourcewithTargetDataMAPU(self):
       
        self.evaluateSourceMAPU(self.trg_train_dl)
        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        print(f'------------------------')
        print(f'Pseudo-label evaluation:')
        print(f'acc\t:{acc}')
        print(f'f1\t:{f1}')
        print(f'auroc\t:{auroc}')
        print(f'------------------------')

        return acc, f1, auroc

    def calculate_metricsSource(self):
       
        self.evaluateSource(self.src_test_dl) 
        # self.evaluate(self.trg_test_dl)
        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        print(f'acc\t:{acc}')
        print(f'f1\t:{f1}')
        print(f'auroc\t:{auroc}')

        return acc, f1, auroc

    def calculate_risks(self):

        self.evaluate_ori(self.src_test_dl) 
        src_risk = self.loss.item()
        # calculation based target test data
        self.evaluate_ori(self.trg_test_dl) 
        # self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, trg_risk

    def calculate_risksSource(self):
         # calculation based source test data
        self.evaluateSource(self.src_test_dl)
        src_risk = self.loss.item()
        trg_risk = src_risk

        return src_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table
    
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table 