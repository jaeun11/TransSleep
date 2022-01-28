import copy
import math
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Model(nn.Module):
    def __init__(self, x):
        super(Model, self).__init__()
        self.FE = MSNN_Feature_Embedding_Two_Way()
        with torch.no_grad():
            b, l, f, t = self.FE(x).shape
            feature_size = f*t
            embedding_size = int(feature_size/2)
        self.sce = Stage_Confusion_Estimator(f)
        self.bilstm = Context_Encoder(feature_size, embedding_size)    # IP = [B, L, F]
        with torch.no_grad():
            x = self.FE(x)
            x = x.flatten(start_dim=2)
            feature_size2 = self.bilstm(x).shape[-1]
        self.project_f = nn.Linear(feature_size, feature_size2)
        self.dropout = nn.Dropout()
        if args.aux_st: self.cls_st = Classifier(feature_size2, 2)
        self.cls = Classifier(feature_size2, 5)


class Stage_Confusion_Estimator(nn.Module):
    def __init__(self, feature_size):
        '''
        Classify FE feature
        '''
        super(Stage_Confusion_Estimator, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.cls_1st = Classifier(feature_size, 5)
        self.project_att = nn.Linear(5, feature_size)

    def forward(self, x):
        b, l, f, t = x.shape
        x = x.view(-1, f, t)
        h = self.gap(x).squeeze()
        l_1 = self.cls_1st(h)
        w = nn.Sigmoid()(self.project_att(F.softmax(l_1, dim=-1))).unsqueeze(-1)
        x_att = x * w
        x_att = x_att.view(b, l, -1)
        l_1 = l_1.view(b, l, -1)
        return x_att, l_1

class Classifier(nn.Module):
    def __init__(self, in_f, out_f):
        '''
        Classify FE feature
        '''
        super(Classifier, self).__init__()
        self.linear_1 = nn.Linear(in_f, out_f)

    def forward(self, x):
        x = self.linear_1(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=args.seq_length):
        '''
        Input shape = [L, B, F] [sequence, batch, d_model(=feature)]
        Output shape = [L, 1, F]
        Only Positional Information
        'M_O': Directional
        'M_M': Symmetric on the Middle Sample
        From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if args.scheme == 'M_O': max_len = max_len//2+1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)    # [seq_length, 1, d_model]
        if args.scheme == 'M_O': pe = torch.cat([pe, pe.flip(dims=(0,1))[1:]])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:x.size(0), :]
        return self.dropout(x)

class MSNN_Feature_Embedding_Two_Way(nn.Module):
    def __init__(self):
        super(MSNN_Feature_Embedding_Two_Way, self).__init__()
        # Define Conv, SepConv
        conv = lambda in_f, out_f, kernel, s=None: nn.Sequential(nn.Conv1d(in_f, out_f, (kernel,), stride=s), nn.BatchNorm1d(out_f), nn.LeakyReLU())
        sepconv_same = lambda in_f, out_f, kernel: nn.Sequential(nn.Conv1d(in_f, out_f, (kernel,), padding=int(kernel/2), groups=in_f),
                                                            nn.Conv1d(out_f, out_f, (1,)), nn.BatchNorm1d(out_f), nn.LeakyReLU())

        self.conv_T_0 = conv(1, 4, 50, args.stride)
        self.sepconv_T_1 = sepconv_same(4, 16, 15)
        self.sepconv_T_2 = sepconv_same(16, 32, 9)
        self.sepconv_T_3 = sepconv_same(32, 64, 5)

        self.conv_S_0 = conv(1, 4, 200, args.stride)
        self.sepconv_S_1 = sepconv_same(4, 16, 11)
        self.sepconv_S_2 = sepconv_same(16, 32, 7)
        self.sepconv_S_3 = sepconv_same(32, 64, 3)

        self.gap = nn.AdaptiveAvgPool1d(args.mha_length)
        self.pe = PositionalEncoding(112, max_len=args.mha_length)
        self.mha_ff_T = nn.TransformerEncoderLayer(112, args.mha_head, dim_feedforward=448)
        self.mha_ff_S = nn.TransformerEncoderLayer(112, args.mha_head, dim_feedforward=448)

    def seq_trans(self, func, x):    # Input shape: [B, F, T]
        return func(x.permute(-1, 0, 1)).permute(1, -1, 0)

    def one_way(self, conv_0, sepconv_1, sepconv_2, sepconv_3, x, mha_ff=None):
        b, l, t = x.shape
        x = x.view(-1, t).unsqueeze(1)    # [B*L, 1, T]
        x = conv_0(x)
        x = sepconv_1(x)
        x1 = x
        x = sepconv_2(x)
        x2 = x
        x = sepconv_3(x)
        x3 = x

        x = self.gap(torch.cat([x1, x2, x3], 1))    # [B, F, T]
        x = x + self.seq_trans(self.pe, x)    # add positional information
        x = x.permute(-1, 0, 1)    # [L, B, F]
        x = mha_ff(x).permute(1, -1, 0)

        x = x.reshape(b, l, *x.shape[-2:])
        return x

    def forward(self, x):
        x_T = self.one_way(self.conv_T_0, self.sepconv_T_1, self.sepconv_T_2, self.sepconv_T_3, x, mha_ff=self.mha_ff_T)
        x_S = self.one_way(self.conv_S_0, self.sepconv_S_1, self.sepconv_S_2, self.sepconv_S_3, x, mha_ff=self.mha_ff_S)
        x = torch.cat((x_T, x_S), dim=-2)    # [B, L, F]
        return x

class Context_Encoder(nn.Module):
    def __init__(self, f, h):
        '''
        Temporal Encoder [Qu et al., 2020]
        Transformer
        '''
        super(Context_Encoder, self).__init__()
        if args.lstm_layers == 1: self.biLSTM = nn.LSTM(f, h, num_layers=args.lstm_layers, bidirectional=True)
        else: self.biLSTM = nn.LSTM(f, h, num_layers=args.lstm_layers, dropout=0.5, bidirectional=True)
        self.dropout_1 = nn.Dropout()
        self.dropout_2 = nn.Dropout()

    def forward(self, x):    # [B, L, F]
        h, _ = self.biLSTM(x.transpose(0, 1))
        h = self.dropout_1(h.transpose(0, 1))
        return h

