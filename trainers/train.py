import sys

# sys.path.append('../ADATIME')
sys.path.append('/home/furqon/blackbox/foundation/mapu5/MAPU_SFDA_TS')

import torch
import os
import pandas as pd

import collections
import argparse
import warnings
import sklearn.exceptions
import numpy as np

from utils import fix_randomness, starting_logs, AverageMeter, starting_logs_source, starting_logs_multi, plot_tsne, plot_tsne2
from abstract_trainer import AbstractTrainer
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()

from moment.momentfm.models.moment import MOMENTPipeline


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}")
        print(f'self.experiment_description:{self.experiment_description}') # self.experiment_description = self.dataset
        self.source_model_dir = os.path.join(self.home_path, self.save_source_dir, self.experiment_description,
                                        f"{self.run_description}", "source")
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def train(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)

                
                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_refine_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id, run_id)

                # Train model
                non_adapted_model, last_adapted_model, best_adapted_model = self.train_model()

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model, last_adapted_model, best_adapted_model)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

    def trainSource(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)
        self.da_method = self.da_method+"_source"
        # print(f'da_method:{da_method}')
        exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}", "source")

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, exp_log_dir,
                                                                   src_id, trg_id, run_id)

                

                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id, run_id)

                # Train model
                non_adapted_model = self.train_model_source()

                # Save checkpoint
                self.save_checkpointSourceOnly(self.home_path, self.scenario_log_dir, non_adapted_model)

                # Calculate risks and metrics
                metrics = self.calculate_metricsSource()
                risks = self.calculate_risksSource()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

    def trainTarget(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        self.da_method = self.da_method+"_target"
        exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}", "target")

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, exp_log_dir,
                                                                   src_id, trg_id, run_id)

                

                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_refine_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_refine_avg_meters2 = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id, run_id)

                self.load_source_model_path = os.path.join(self.source_model_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))

                # Train model
                model1, model2 = self.train_model_target() # COT

                # Save checkpoint
                self.save_checkpointTargetCOT(self.home_path, self.scenario_log_dir, model1, model2)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

    

    def trainSourceMulti(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        
        self.da_method = self.da_method+"_source"
        # print(f'da_method:{da_method}')
        exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}", "source")
        
        self.source_models = {}

        # For convenience, grab the list of source IDs once
        all_src_ids = self.dataset_configs.src_domains  # e.g. ["2","6","7","9","12"]

        # Loop over each target domain
        for run_id in range(self.num_runs):
            fix_randomness(run_id)

            # For each source domain, build & train a separate model
            for src_id in all_src_ids:
                # Logging setup
                self.logger, self.scenario_log_dir = starting_logs_source(
                    self.dataset,
                    f"{self.da_method}_per_domain",     # or any label you like
                    exp_log_dir,
                    src_id = src_id,               
                    run_id      = run_id
                )

                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.load_data_source(src_id=src_id, run_id=run_id)

                # Train model
                non_adapted_model = self.train_model_source_multi()

                # Save checkpoint
                self.save_checkpointSourceOnly(self.home_path, self.scenario_log_dir, non_adapted_model)

                # Calculate risks and metrics
                metrics = self.calculate_metricsSource()

                # Append results to tables
                scenario = f"{src_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')

    

    def trainTargetMulti(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]

        self.da_method = self.da_method+"_target_multi"
        exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}", "target_multi")
        all_src_ids = self.dataset_configs.src_domains

        # Trainer
        for trg_id in self.dataset_configs.trg_domains:
            for run_id in range(self.num_runs):
                fix_randomness(run_id)
                src_ids = " ".join(all_src_ids)
                # Logging
                self.logger, self.scenario_log_dir = starting_logs_multi(self.dataset, self.da_method, exp_log_dir,
                                                                   src_ids, trg_id, run_id)
                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data_target(trg_id, run_id)

                # Source model path
                self.load_source_model_path = []
                for s in all_src_ids:
                    source_path = os.path.join(self.source_model_dir, s + "_run_" + str(run_id))
                    self.load_source_model_path.append(source_path)

                # Train model
                model1, model2 = self.train_model_target_multi() # COT

                # Save checkpoint
                self.save_checkpointTargetCOT(self.home_path, self.scenario_log_dir, model1, model2)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()

                # Append results to tables
                scenario = f"{src_ids}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')

    


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs_Mapu', type=str,
                        help='Directory containing all experiments')

    
    parser.add_argument('--save_source_dir', default='HAR_source_run3_dropLast_false', type=str, # HAR 25 Juni 2025
                        help='Directory containing all experiments')
    
    parser.add_argument('--save_target_dir', default='HAR_target_run3_dropLast_false', type=str, # DUAL 25 Juni 2025
                        help='Directory containing all experiments')


    parser.add_argument('-run_description', default=None, type=str, help='Description of run, if none, DA method name will be used')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='B2TSDA_COT', type=str, help='SHOT, AaD, NRC, MAPU, B2TSDA')


    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'/home/furqon/mapu/MAPU_SFDA_TS/data/HAR Dataset', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')


    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='Transformer', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_neighbors', default=10, type=int)
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--plot_tsne', default=True, type=bool, help='Plot t-sne for training and testing or not?')

    args = parser.parse_args()

    trainer = Trainer(args)
    # trainer.trainSource()
    trainer.trainTarget()
    # trainer.trainSourceMulti()
    # trainer.trainTargetMulti()
