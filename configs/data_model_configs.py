import os
import torch


def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        # self.src_domains = ['0', '7', '9', '12', '16']
        self.src_domains = ['0', '7', '9']
        # self.src_domains = ['0']
        # self.src_domains = ['7']
        # self.src_domains = ['9']
        # self.src_domains = ['12']
        # self.src_domains = ['16']
        # self.src_domains = ['0', '7', '9']
        self.trg_domains = ['1', '5', '11', '14', '18']
        # self.trg_domains = ['5', '11', '14', '18']
        # self.trg_domains = ['1']
        # self.trg_domains = ['5']
        # self.trg_domains = ['11']
        # self.trg_domains = ['14']
        # self.trg_domains = ['18']
        # self.scenarios = [("9", "14")]
        self.shuffle = True
        self.drop_last = True #True
        self.normalize = True
        self.noise_rate = 0.2

        # self.jitter_scale_ratio = 0.5 # 13 Nov
        # self.jitter_ratio = 2
        # self.max_seg = 12

        # self.jitter_scale_ratio = 0.05 # 20 Nov
        # self.jitter_ratio = 2.4
        # self.max_seg = 12

        # self.jitter_scale_ratio = 0.08 # 11 Maret
        # self.jitter_ratio = 2.5
        # self.max_seg = 12

        self.jitter_scale_ratio = 0.005
        self.jitter_ratio = 2.5
        self.max_seg = 12

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        # self.dropout = 0.2
        self.dropout_src = 0.0000005
        self.dropout = 0.0000005 # 4 Nov
        # self.dropout = 0.005 
        # self.dropout = 0.000005 # HugFace
        # self.prompt_length = 40
        self.prompt_length = 5 
        # self.prompt_length = 40 # 6 Nov
        # self.prompt_length = 24

        # features
        self.mid_channels = 16
        self.final_out_channels = 8
        self.features_len = 65 # for my model
        self.AR_hid_dim = 8

        # auto encoder
        self.mid_dim = 375
        self.out_dim = 250
        self.TSlength_aligned = 1024

        # AR Discriminator
        self.disc_hid_dim = 256
        self.disc_AR_bid= False
        self.disc_AR_hid = 128
        self.disc_n_layers = 1
        self.disc_out_dim = 1
class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        # self.scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0"), ("2", "3")]
        self.scenarios = [("2", "3")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.src_domains = ['0', '1', '2', '3']
        # self.src_domains = ['0', '1']
        self.trg_domains = ['0', '1', '2', '3']
        # self.trg_domains = ['0']
        # self.trg_domains = ['1']
        # self.trg_domains = ['2']
        # self.trg_domains = ['3']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        self.jitter_scale_ratio = 0.005
        self.jitter_ratio = 2.5
        self.max_seg = 12

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5
        # self.dropout = 0.0000005 # 4 Nov
        self.dropout_src = 0.0000005
        self.prompt_length = 5 

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # auto encoder
        self.mid_dim = 375
        self.out_dim = 250
        self.TSlength_aligned = 1024

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.AR_hid_dim = 128
class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"),  ]
        # self.scenarios = [("7", "13")]

        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.src_domains = ['2', '6', '7', '9', '12']
        # self.src_domains = ['2']
        # self.src_domains = ['2', '6', '7']
        # self.src_domains = ['9', '12']
        self.trg_domains = ['11', '13', '16', '18', '23']
        # self.trg_domains = ['11']
        # self.trg_domains = ['13']
        # self.trg_domains = ['16']
        # self.trg_domains = ['18']
        # self.trg_domains = ['23']
        self.sequence_len = 128
        self.shuffle = True
        # self.drop_last = True #False
        self.drop_last = False
        self.normalize = True

        self.jitter_scale_ratio = 0.005
        self.jitter_ratio = 2.5
        self.max_seg = 12

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.0000005
        self.num_classes = 6
        self.dropout_src = 0.0000005
        self.prompt_length = 5

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # self.features_len = 18 for sequential methods
        self.AR_hid_dim = 128

        # auto encoder
        self.mid_dim = 375 #asli
        self.out_dim = 250 #asli
        # self.mid_dim = 256
        # self.out_dim = 128
        self.TSlength_aligned = 1024




