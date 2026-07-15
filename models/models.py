
import copy
import torch
import math
from torch import nn
from einops import rearrange
from collections import OrderedDict
from utils import _normalize_t
from models.prompt import Prompt
import torch.nn.functional as F


def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


## Feature Extractor
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = self.aap(x).view(x.shape[0], -1)

        return x_flat, x

## Feature Bootleneck
class feat_bootleneck(nn.Module):
    def __init__(self, configs, bottleneck_dim=256, type="bn"):   #type="ori"
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        # self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck = nn.Linear(configs.features_len, bottleneck_dim)
        # self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn" or self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.bn(x)
        if self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.relu(x)
        if self.type == "bn_relu_drop":
            x = self.dropout(x)
        return x


# temporal masking
def masking(x, num_splits=8, num_masked=4):
    # num_masked = int(masking_ratio * num_splits)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    masked_patches = patches.clone()  # deepcopy(patches)
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    masks = []
    for i in range(masked_patches.shape[1]):
        masks.append(masked_patches[:, i, (selected_indices[i, :]), :])
        masked_patches[:, i, (selected_indices[i, :]), :] = 0
        # orig_patches[:, i, (selected_indices[i, :]), :] =
    mask = rearrange(torch.stack(masks), 'b a p l -> a b (p l)')
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask

class AutoEncoder(nn.Module):
    def __init__(self, dim, decoder_dim, inner_dim):
        super().__init__()
        self.prompt_w1 = nn.Linear(dim, inner_dim)
        self.prompt_w2 = nn.Linear(inner_dim, decoder_dim * 2)  # Increased decoder capacity
        self.prompt_w3 = nn.Linear(decoder_dim * 2, dim)  # Ensure output size matches input

    def forward(self, x):
        x = x.to(self.prompt_w1.weight.device)
        
        # Normalize input (stabilizes training)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        
        x = self.prompt_w1(x)
        x = torch.tanh(self.prompt_w2(x))
        x = self.prompt_w3(x)
        
        # Force output range to match input distribution
        x = torch.tanh(x)  
        
        return x


# Domain Classifier
class DomainClassifier(nn.Module):
    def __init__(self, feature_dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification (source=0, target=1)

    def forward(self, features):
        if features.dim() == 3:  # If features are [batch_size, n_patches, feature_dim]
            features = torch.mean(features, dim=1)  # Apply mean pooling

        x = F.relu(self.fc1(features))
        return self.fc2(x)  # No softmax, as we use BCEWithLogitsLoss


# Transformer
class Transformer(nn.Module):
    # def __init__(self, in_dim=1, out_dim=128, n_layer=8, n_dim=64, n_head=8,
    #              norm_first=False, is_pos=True, is_projector=True,
    #              project_norm=None, dropout=0.0):
    def __init__(self, config, conf_dropout, in_dim=1, out_dim=128, n_layer=4, n_dim=64, n_head=8,
                 norm_first=False, is_pos=True, is_projector=True,
                 project_norm='LN', dropout=0.0):
        r"""
        Transformer-based time series encoder

        Args:
            in_dim (int, optional): Number of dimension for the input time
                series. Default: 1.
            out_dim (int, optional): Number of dimension for the output
                representation. Default: 128.
            n_layer (int, optional): Number of layer for the transformer
                encoder. Default: 8.
            n_dim (int, optional): Number of dimension for the intermediate
                representation. Default: 64.
            n_head (int, optional): Number of head for the transformer
                encoder. Default: 8.
            norm_first: if ``True``, layer norm is done prior to attention and
                feedforward operations, respectively. Otherwise it's done
                after. Default: ``False`` (after).
            is_pos (bool, optional): If set to ``False``, the encoder will
                not use position encoding. Default: ``True``.
            is_projector (bool, optional): If set to ``False``, the encoder
                will not use additional projection layers. Default: ``True``.
            project_norm (string, optional): If set to ``BN``, the projector
                will use batch normalization. If set to ``LN``, the projector
                will use layer normalization. If set to None, the projector
                will not use normalization. Default: None (no normalization).
            dropout (float, optional): The probability of an element to be
                zeroed for the dropout layers. Default: 0.0.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`, :math:`(N, L_{in})`, or
                :math:`(L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(Transformer, self).__init__()
        assert project_norm in ['BN', 'LN', None]

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_dim = n_dim
        self.is_projector = is_projector
        self.is_pos = is_pos
        self.max_len = 0
        self.dropout = conf_dropout
        # print(f'self.dropout:{self.dropout}')

        # self.prompt = Prompt(length=config.prompt_length, embed_dim=n_dim, prompt_init='uniform')

        self.in_net = nn.Conv1d(
            in_dim, n_dim, 7, stride=2, padding=3, dilation=1)
        self.add_module('in_net', self.in_net)
        transformer = OrderedDict()
        for i in range(n_layer):
            transformer[f'encoder_{i:02d}'] = nn.TransformerEncoderLayer(
                n_dim, n_head, dim_feedforward=n_dim,
                dropout=dropout, batch_first=True,
                norm_first=norm_first)
        self.transformer = nn.Sequential(transformer)

        self.start_token = nn.Parameter(
            torch.randn(1, n_dim, 1))
        self.register_parameter(
            name='start_token',
            param=self.start_token)

        self.out_net = nn.Linear(n_dim, out_dim)
        self.project_norm = project_norm
        if is_projector:
            if project_norm == 'BN':
                self.projector = nn.Sequential(
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.BatchNorm1d(out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim)
                )
            elif project_norm == 'LN':
                self.projector = nn.Sequential(
                    nn.ReLU(),
                    nn.LayerNorm(out_dim),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(out_dim * 2),
                    nn.Linear(out_dim * 2, out_dim)
                )
            else:
                self.projector = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim)
                )
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False):
        device = self.dummy.device
        is_projector = self.is_projector
        is_pos = self.is_pos

        ts = _normalize_t(ts, normalize)
        ts = ts.to(device, dtype=torch.float32)

        ts_emb = self.in_net(ts)
        if is_pos:
            n_dim = self.n_dim
            dropout = self.dropout
            ts_len = ts_emb.size()[2]
            if ts_len > self.max_len:
                self.max_len = ts_len
                self.pos_net = PositionalEncoding(
                    n_dim, ts_len, dropout=dropout)
                self.pos_net.to(device)
            ts_emb = self.pos_net(ts_emb)

        start_tokens = self.start_token.expand(ts_emb.size()[0], -1, -1)
        ts_emb = torch.cat((start_tokens, ts_emb, ), dim=2)
        ts_emb = torch.transpose(ts_emb, 1, 2)

        ts_emb = self.transformer(ts_emb)
        ts_emb = ts_emb[:, 0, :]
        ts_emb = self.out_net(ts_emb)

        if is_projector:
            ts_emb = self.projector(ts_emb)

        if to_numpy:
            return ts_emb.cpu().detach().numpy()
        else:
            return ts_emb

    def encode(self, ts, normalize=True, to_numpy=False):
        return self.forward(ts, normalize=normalize, to_numpy=to_numpy)

    def encode_seq(self, ts, normalize=True, to_numpy=False):
        device = self.dummy.device
        is_projector = self.is_projector
        is_pos = self.is_pos

        ts = _normalize_t(ts, normalize)
        ts = ts.to(device, dtype=torch.float32)

        ts_emb = self.in_net(ts)
        if is_pos:
            n_dim = self.n_dim
            dropout = self.dropout
            ts_len = ts_emb.size()[2]
            if ts_len > self.max_len:
                self.max_len = ts_len
                self.pos_net = PositionalEncoding(
                    n_dim, ts_len, dropout=dropout)
                self.pos_net.to(device)
            ts_emb = self.pos_net(ts_emb)

        start_tokens = self.start_token.expand(ts_emb.size()[0], -1, -1)
        ts_emb = torch.cat((start_tokens, ts_emb, ), dim=2)
        ts_emb = torch.transpose(ts_emb, 1, 2)

        ts_emb = self.transformer(ts_emb)
        ts_emb = self.out_net(ts_emb)
        if is_projector:
            project_norm = self.project_norm
            if project_norm == 'BN':
                layers = [module for module in is_projector.modules()]
                ts_emb = torch.transpose(ts_emb, 1, 2)
                ts_emb = layers[1](ts_emb)
                ts_emb = torch.transpose(ts_emb, 1, 2)
                ts_emb = layers[2](ts_emb)
                ts_emb = layers[3](ts_emb)
                ts_emb = torch.transpose(ts_emb, 1, 2)
                ts_emb = layers[4](ts_emb)
                ts_emb = torch.transpose(ts_emb, 1, 2)
                ts_emb = layers[5](ts_emb)
                ts_emb = layers[6](ts_emb)
            else:
                ts_emb = self.projector(ts_emb)

        ts_emb = ts_emb[:, 1:, :]
        start_tokens = ts_emb[:, 0:1, :]
        start_tokens = start_tokens.expand(-1, ts_emb.size()[1], -1)
        ts_emb = ts_emb + start_tokens
        ts_emb = torch.transpose(ts_emb, 1, 2)

        if to_numpy:
            return ts_emb.cpu().detach().numpy()
        else:
            return ts_emb

    def freeze_parameters(model):
        """
        Freeze parameters of the model
        """
        # Freeze the parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        return model


class PositionalEncoding(nn.Module):
    def __init__(self, n_dim, max_len, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len)
        div_term = torch.exp(
            torch.arange(0, n_dim, 2) * (-math.log(10000.0) / n_dim))
        pos_emb = torch.zeros(1, n_dim, max_len)

        position = position.unsqueeze(0)
        div_term = div_term.unsqueeze(1)
        pos_emb[0, 0::2, :] = torch.sin(div_term * position)
        pos_emb[0, 1::2, :] = torch.cos(div_term * position)
        self.register_buffer('pos_emb', pos_emb, persistent=False)

    def forward(self, x):
        x = x + self.pos_emb[:, :, :x.size()[2]]
        return self.dropout(x)

class Classifier(nn.Module):
    def __init__(self, encoder, n_class, n_dim=64, n_layer=2):
        super(Classifier, self).__init__()
        self.encoder = encoder
        # print(f'self.encoder:{self.encoder}')
        # self.add_module('encoder', encoder)

        in_dim_ = self.encoder.out_dim
        out_dim_ = n_dim
        layers = OrderedDict()
        for i in range(n_layer - 1):
            layers[f'linear_{i:02d}'] = nn.Linear(
                in_dim_, out_dim_)
            layers[f'relu_{i:02d}'] = nn.ReLU()
            in_dim_ = out_dim_
            out_dim_ = n_dim

        layers[f'linear_{n_layer - 1:02d}'] = nn.Linear(
            in_dim_, n_class)
        self.classifier = nn.Sequential(layers)

    def forward(self, ts, normalize=True, to_numpy=False):
        hidden = self.encoder.encode(
            ts, normalize=normalize, to_numpy=False)
        logit = self.classifier(hidden)
        if to_numpy:
            return logit.cpu().detach().numpy()
        else:
            return logit

class TimeCLREncoder(nn.Module):
    def __init__(self, encoder, aug_bank):
        r"""
        The proposed TimeCLR method

        Args:
            encoder (Module): The base encoder
            aug_bank (list): A list of augmentation methods.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(TimeCLREncoder, self).__init__()

        self.pretrain_name = 'timeclr'
        self.encoder = copy.deepcopy(encoder)

        self.aug_bank = aug_bank
        n_aug = len(aug_bank)
        self.n_aug = n_aug

        self.out_dim = self.encoder.out_dim
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False, is_augment=False):
        if is_augment:
            ts = self._augment_ts(ts)

        ts_emb = self.encoder.encode(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb

    def encode(self, ts, normalize=True, to_numpy=False):
        ts_emb = self.encoder.encode(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb

    def _augment_ts(self, ts):
        n_ts = ts.shape[0]
        n_aug = self.n_aug
        ts_aug = copy.deepcopy(ts)
        aug_bank = self.aug_bank
        for i in range(n_ts):
            aug_idx = np.random.randint(n_aug)
            ts_aug[i, 0, :] = aug_bank[aug_idx](ts_aug[i, 0, :])
        return ts_aug
