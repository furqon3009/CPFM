def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            # 'num_epochs': 40,
            'num_iter':20,
            'num_epochs': 20,
            'tau': 0.1,
            # 'tau': 0.009,
            # 'lam': 0.5,
            'lam': 1.5,
            'threshold': 0.8,
            'gma': 1, # last setting
            # 'gma': 0.85,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 0},
            'AaD': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'beta': 5, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'B2TSDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'B2TSDA_COT': {'pre_learning_rate':  0.001, 'learning_rate': 0.01, 'learning_rate_refine': 0.0001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 
                        'TOV_wt': 0.6385, 'forget_rate': 0, 'warm_target': 5, 'num_gradual': 5, 'learning_rate_AE': 0.01,},
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            # 'num_epochs': 40,
            # 'num_epochs': 100,
            'num_epochs': 20,
            # 'num_epochs': 5,
            'num_iter':20,
            # 'num_epochs': 60,
            'tau': 0.1,
            # 'tau': 0.009,
            # 'lam': 0.5,
            'lam': 1.5,
            'threshold': 0.8,
            'gma': 1, # last setting
            # 'gma': 0.85,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.0081},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'beta': 9, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'MAPU2': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'learning_rate_refine': 0.001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'DINE2': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385, 'topk': 1, 'ema': 0.6},
            # 'B2TSDA': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            # 'B2TSDA': {'pre_learning_rate':  0.003, 'learning_rate': 0.1, 'ent_loss_wt': 0.4216, 'im': 0.5514, 
            #             'TOV_wt': 0.6385, 'forget_rate': 0.4, 'warm_target': 5, 'num_gradual': 5, 'learning_rate_AE': 0.001,},
            # 'B2TSDA': {'pre_learning_rate':  0.003, 'learning_rate': 0.01, 'ent_loss_wt': 0.4216, 'im': 0.5514, 
            #             'TOV_wt': 0.6385, 'forget_rate': 0.4, 'warm_target': 5, 'num_gradual': 5, 'learning_rate_AE': 0.001,},
            # 'B2TSDA_CEOnly': {'pre_learning_rate':  0.003, 'learning_rate': 0.1, 'ent_loss_wt': 0.4216, 'im': 0.5514, 
            #             'TOV_wt': 0.6385, 'forget_rate': 0.4, 'warm_target': 5, 'num_gradual': 5, 'learning_rate_AE': 0.001,},
            # 'B2TSDA_NoCOT': {'pre_learning_rate':  0.05, 'learning_rate': 0.001, 'learning_rate_refine': 0.001, 'ent_loss_wt': 0.4216, 'im': 0.5514,  # HugFace
            #             'TOV_wt': 0.6385, 'learning_rate_AE': 0.001},
            'B2TSDA_NoCOT': {'pre_learning_rate':  0.003, 'learning_rate': 0.001, 'learning_rate_refine': 0.001, 'ent_loss_wt': 0.4216, 'im': 0.5514, # 4 Nov
                        'TOV_wt': 0.6385, 'learning_rate_AE': 0.1},
            'B2TSDA_Only': {'pre_learning_rate':  0.03, 'learning_rate': 0.01, 'learning_rate_refine': 0.01, 'ent_loss_wt': 0.4216, 'im': 0.5514, 
                        'TOV_wt': 0.6385, 'learning_rate_AE': 0.001},
            # 'B2TSDA_COT': {'pre_learning_rate':  0.003, 'learning_rate': 0.01, 'learning_rate_refine': 0.001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 
            #             'TOV_wt': 0.6385, 'learning_rate_AE': 0.001},
            # 'B2TSDA_COT': {'pre_learning_rate':  0.003, 'learning_rate': 0.01, 'learning_rate_refine': 0.01, 'ent_loss_wt': 0.4216, 'im': 0.5514,  # 6 maret (40 epoch)
            #             'TOV_wt': 0.6385, 'forget_rate': 0, 'warm_target': 5, 'num_gradual': 5, 'learning_rate_AE': 0.001,},
            'B2TSDA_COT': {'pre_learning_rate':  0.003, 'learning_rate': 0.01, 'learning_rate_refine': 0.0001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 
                        'TOV_wt': 0.6385, 'forget_rate': 0, 'warm_target': 5, 'num_gradual': 5, 'learning_rate_AE': 0.01,},

        }


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            # 'num_epochs': 40, # for source model
            'num_epochs': 20,
            'num_iter':20,
            'tau': 0.1,
            # 'tau': 0.009,
            # 'lam': 0.5,
            'lam': 1.5,
            'threshold': 0.8,
            'gma': 1, # last setting
            # 'gma': 0.7, 
            'batch_size': 32, # for single source
            # 'batch_size': 16, # for multi-source
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'alpha': 0.9
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.6709, 'im': 0.8969,
                     'target_cls_wt': 0.3312},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1},

            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'B2TSDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'B2TSDA_COT': {'pre_learning_rate':  0.001, 'learning_rate': 0.01, 'learning_rate_refine': 0.0001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 
                        'TOV_wt': 0.6385, 'forget_rate': 0, 'warm_target': 5, 'num_gradual': 5, 'learning_rate_AE': 0.01,},
        }


