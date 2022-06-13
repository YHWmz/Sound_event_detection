import math

import torch.nn as nn
import torch
from torch.nn import functional as F
from conformer import Conformer
from conformer.encoder import ConformerBlock
from conformer.feed_forward import FeedForwardModule
from conformer.attention import MultiHeadedSelfAttentionModule
from conformer.convolution import (
    ConformerConvModule,
    Conv2dSubampling,
)
from conformer.modules import (
    ResidualConnectionModule,
    Linear,
)
from conformer.activation import Swish
from conformer.modules import Linear, Transpose

def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)

def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)
    
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        self.maxpool = torch.nn.MaxPool2d((2,2))
        self.dropout = torch.nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn(self.conv(x))))
        return self.dropout(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 dilation=1, bias=False):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.double_conv:
            init_layer(layer)

    def forward(self, x):
        x = self.double_conv(x)

        return x

class Crnn(nn.Module):
    def __init__(self, inputdim, outputdim, dropout=0.0):
        super().__init__()
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        # self.bn = torch.nn.BatchNorm1d(num_freq)

        # self.conv1 = DoubleConv(1, 16)
        # self.conv2 = DoubleConv(16, 32)
        # self.conv3 = DoubleConv(32, 64)
        self.conv1 = ConvBlock(1, 16, dropout)
        self.conv2 = ConvBlock(16, 32, dropout)
        self.conv3 = ConvBlock(32, 64, dropout)

        self.gru = torch.nn.GRU(input_size=64, hidden_size=64,
                            num_layers=1, batch_first=True,
                            dropout=dropout, bidirectional=True)

        self.fc = torch.nn.Linear(128, outputdim)
        self.sigmoid = torch.nn.Sigmoid()


    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        # []
        N, T, _ = x.shape
        # x = x.permute(0, 2, 1)
        # x = self.bn(x).permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.mean(dim=3).permute(0, 2, 1)
        (x, _) = self.gru(x)

        x = self.fc(x)
        x = self.sigmoid(x)
        x = F.interpolate(x.transpose(1, 2), T).transpose(1, 2)
        return x


    def forward(self, x):
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }

###################################################################3

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MaxPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.max(decision, dim=self.pooldim)[0]


class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class AutoExpPool(nn.Module):
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.full((outputdim, ), 1))
        self.pooldim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        return (logits * torch.exp(scaled)).sum(
            self.pooldim) / torch.exp(scaled).sum(self.pooldim)


class SoftPool(nn.Module):
    def __init__(self, T=1, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.T = T

    def forward(self, logits, decision):
        w = torch.softmax(decision / self.T, dim=self.pooldim)
        return torch.sum(decision * w, dim=self.pooldim)


class AutoPool(nn.Module):
    """docstring for AutoPool"""
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.ones(outputdim))
        self.dim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        weight = torch.softmax(scaled, dim=self.dim)
        return torch.sum(decision * weight, dim=self.dim)  # B x C


class ExtAttentionPool(nn.Module):
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.attention = nn.Linear(inputdim, outputdim)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.activ = nn.Softmax(dim=self.pooldim)

    def forward(self, logits, decision):
        # Logits of shape (B, T, D), decision of shape (B, T, C)
        w_x = self.activ(self.attention(logits) / self.outputdim)
        h = (logits.permute(0, 2, 1).contiguous().unsqueeze(-2) *
             w_x.unsqueeze(-1)).flatten(-2).contiguous()
        return torch.sum(h, self.pooldim)


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1
    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'max':
        return MaxPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'expalpha':
        return AutoExpPool(outputdim=kwargs['outputdim'], pooldim=1)

    elif poolingfunction_name == 'soft':
        return SoftPool(pooldim=1)
    elif poolingfunction_name == 'auto':
        return AutoPool(outputdim=kwargs['outputdim'])
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])


class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

    def forward(self, x):
        return self.block(x)

class CDur_sim(nn.Module):
    def __init__(self, inputdim=64, outputdim=10, **kwargs):
        super().__init__()
        features = nn.ModuleList()
        self.features = nn.Sequential(
            Block2D(1, 32),

            # nn.MaxPool2d((2, 4)),
            # nn.AvgPool2d((2, 4)),
            nn.LPPool2d(4, (2, 4)),

            Block2D(32, 128),
            Block2D(128, 128),
            # Conv_standard(in_channels=32, out_channels=128),
            # Conv_standardPOST_ELU(in_channels=32, out_channels=128),
            # Conv_standardPOST(in_channels=32, out_channels=128),

            # nn.MaxPool2d((2, 4)),
            # nn.AvgPool2d((2, 4)),
            nn.LPPool2d(4, (2, 4)),

            Block2D(128, 128),
            Block2D(128, 128),
            # Conv_standard(in_channels=128, out_channels=128),
            # Conv_standardPOST_ELU(in_channels=128, out_channels=128),
            # Conv_standardPOST(in_channels=128, out_channels=128),

            # nn.MaxPool2d((1, 4)),
            # nn.AvgPool2d((1, 4)),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        # self.trans = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=rnn_input_dim, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=1)
        # self.trans = ConformerBlock(
        #     encoder_dim=rnn_input_dim,
        #     num_attention_heads=8,
        #     feed_forward_expansion_factor=4,
        #     conv_expansion_factor=2,
        #     feed_forward_dropout_p=0.2,
        #     attention_dropout_p=0.2,
        #     conv_dropout_p=0.2,
        #     conv_kernel_size=31,
        #     half_step_residual=True,
        # )
        # self.downfc = nn.Linear(rnn_input_dim, 256)
        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          num_layers=1,
                          bidirectional=True,
                          batch_first=True)
        # mean max linear expalpha attention
        self.temp_pool = parse_poolingfunction('linear',
                                               inputdim=256,
                                               outputdim=outputdim)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        # x = self.downfc(self.trans(x))
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        if upsample:
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': decision,
            'time_probs': decision_time
        }
        # return decision, decision_time

class CDur(nn.Module):
    def __init__(self, inputdim=64, outputdim=10, **kwargs):
        super().__init__()
        features = nn.ModuleList()
        self.features = nn.Sequential(
            Block2D(1, 32),

            # nn.MaxPool2d((2, 4)),
            # nn.AvgPool2d((2, 4)),
            nn.LPPool2d(4, (2, 4)),

            Block2D(32, 128),
            Block2D(128, 128),

            # nn.MaxPool2d((2, 4)),
            # nn.AvgPool2d((2, 4)),
            nn.LPPool2d(4, (2, 4)),

            Block2D(128, 128),
            Block2D(128, 128),

            # nn.MaxPool2d((1, 4)),
            # nn.AvgPool2d((1, 4)),
            nn.LPPool2d(4, (1, 4)),

            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
                                               inputdim=256,
                                               outputdim=outputdim)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        if upsample:
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': decision,
            'time_probs': decision_time
        }
        # return decision, decision_time

class Conv_standardPOST(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__()

        self.preblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        # self.elu = nn.ELU(inplace=True)
        self.elu = nn.ReLU(inplace=True)
        self.scSE = scSEblock(out_channels)

        self.init_weights()

    def init_weights(self):
        for layer in self.preblock:
            init_layer(layer)
        for layer in self.resblock:
            init_layer(layer)

    def forward(self, x):
        x_res = self.resblock(x)
        x = self.preblock(x)

        x = x + x_res
        x = self.elu(x)
        x = self.scSE(x)

        x = x + x_res

        return x

class Conv_standardPOST_ELU(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__()

        self.preblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            # nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        self.elu = nn.ELU(inplace=True)
        # self.elu = nn.ReLU(inplace=True)
        self.scSE = scSEblock(out_channels)

        self.init_weights()

    def init_weights(self):
        for layer in self.preblock:
            init_layer(layer)
        for layer in self.resblock:
            init_layer(layer)

    def forward(self, x):
        x_res = self.resblock(x)
        x = self.preblock(x)

        x = x + x_res
        x = self.elu(x)
        x = self.scSE(x)

        x = x + x_res
        x = self.elu(x)

        return x

class Conv_POST_ELU(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__()

        self.preblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        # self.elu = nn.ELU(inplace=True)
        self.elu = nn.ReLU(inplace=True)
        self.scSE = scSEblock(out_channels)

        self.init_weights()

    def init_weights(self):
        for layer in self.preblock:
            init_layer(layer)
        for layer in self.resblock:
            init_layer(layer)

    def forward(self, x):
        x_res = self.resblock(x)
        x = self.preblock(x)

        x = x + x_res
        x = self.elu(x)
        x = self.scSE(x)

        return x

class Conv_standard(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__()

        self.preblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        # self.elu = nn.ELU(inplace=True)
        self.elu = nn.ReLU(inplace=True)
        self.scSE = scSEblock(out_channels)

        self.init_weights()

    def init_weights(self):
        for layer in self.preblock:
            init_layer(layer)
        for layer in self.resblock:
            init_layer(layer)

    def forward(self, x):
        x_res = self.resblock(x)
        x = self.preblock(x)

        x = self.scSE(x)

        x = x + x_res

        return x

class cSEblock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels, in_channels//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channels//ratio, in_channels, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        init_layer(self.compress)
        init_layer(self.excitation)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.relu(self.compress(out))
        out = self.sigmoid(self.excitation(out))
        x = x * out
        return x

class sSEblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        init_layer(self.spatial_conv)

    def forward(self, x):
        out = self.sigmoid(self.spatial_conv(x))
        x = x * out
        return x

class scSEblock(nn.Module):
    def __init__(self, channel):
        super(scSEblock, self).__init__()
        self.cse = cSEblock(channel)
        self.sse = sSEblock(channel)

    def forward(self, x):
        x1 = self.cse(x)
        x2 = self.sse(x)
        x = x1 + x2
        return x

