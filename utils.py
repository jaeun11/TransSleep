from config import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch import default_generator  # type: ignore
from typing import Tuple
from torch import Tensor, Generator
import mne
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from numpy.random import shuffle

''' Record File Name'''
def record_name(cv_i):
    # return args.characteristic + '_Finetuning_Model_' + args.finetuning_model_name + str(args.lr_finetune_network)
    return args.characteristic + '_CV_' + str(cv_i)

def writelog(file, line):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line + '\n')

''' Functions '''
def label_stage_transition(label_list, log_file):
    for l in range(len(label_list)):
        label_shape = label_list[l].shape
        label = label_list[l].flatten()
        lbl = np.zeros_like(label)
        for i in range(1, len(label)):
            if i!=len(label)-1:
                if label[i] == label[i - 1] and label[i] == label[i + 1]: lbl[i] = 0
                else: lbl[i] = 1
            else:
                if label[i] == label[i - 1]: lbl[i] = 0
                else: lbl[i] = 1
        cls, count = np.unique(lbl, return_counts=True)
        writelog(log_file, f'Lable Count: {dict(zip(cls, count))}')
        lbl = lbl.reshape(label_shape)
        label_list[l] = lbl
    return label_list

def float_tensor(x):
    return torch.FloatTensor(x)

def long_tensor(x):
    return torch.LongTensor(x)

def dcn(x):    # detach, cpu, numpy
    if type(x)== np.ndarray: return x
    else: return x.detach().cpu().numpy()

def pb_argmax(x):
    if len(x.shape) == 1: return(x)
    else: return np.argmax(x, axis=-1)

def flatten_1dim(x):
    if len(x.shape) == 1: return (x)
    else: return x.flatten()

def flatten_logit_list(l):
    for i in range(len(l)):
        l[i] = l[i].flatten(end_dim=1)
    return l

def standardize(x):
    return (x - x.mean(axis=1)[:, None]) / x.std(axis=1)[:, None]

def downsample_to_100(x):
    x = mne.filter.resample(x, down=2.56, axis=-1)
    return x


def idx_tr_val_split(x):    # TODO: when do we need seed?
    idx_all = np.random.RandomState(seed=961125).permutation(x)
    idx_val = idx_all[:int(x/cv)]    # num of int(x/cv)
    idx_tr = np.setdiff1d(idx_all, idx_val)
    return idx_tr, idx_val

def tr_val_ts_split(idx_sbj_ts, idx_sbj_total, X_dict, Y_dict, log_file):
    X_ts = np.concatenate([X_dict[idx_sbj_ts[i]] for i in range(len(idx_sbj_ts))])
    Y_ts = np.concatenate([Y_dict[idx_sbj_ts[i]] for i in range(len(idx_sbj_ts))])
    idx_sbj_tr = sorted(set(idx_sbj_total)-set(idx_sbj_ts))
    writelog(log_file, f'Idx_sbj_ts: {idx_sbj_ts}\n')
    writelog(log_file, f'Num of Training File: {len(idx_sbj_tr)}\n')
    X_tr_val = np.concatenate([X_dict[i] for i in idx_sbj_tr])
    Y_tr_val = np.concatenate([Y_dict[i] for i in idx_sbj_tr])
    idx_tr, idx_val = idx_tr_val_split(X_tr_val.shape[0])    # Tr | Val
    X_tr, Y_tr, X_val, Y_val = X_tr_val[idx_tr], Y_tr_val[idx_tr], X_tr_val[idx_val], Y_tr_val[idx_val]
    return X_tr, Y_tr, X_val, Y_val, X_ts, Y_ts

def tr_val_split(idx_tr, idx_val, X_tr_val, Y_tr_val):
    return X_tr_val[idx_tr], Y_tr_val[idx_tr], X_tr_val[idx_val], Y_tr_val[idx_val]

def pytorch_sliding_window(x, window_size, step_size=1):
    # Unfold Dimension to Make Slding Window
    return x.unfold(0,window_size,step_size)

''' Count Parameters '''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

''' Loss '''
def loss_cross_entropy(weight=None, reduction='mean'):    # Take Logit as an Input
    return nn.CrossEntropyLoss(weight=weight, reduction=reduction)

def loss_cos_loss(margin=0, reduction='mean'):
    return nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

def loss_mse(reduction='none'):
    return torch.nn.MSELoss(reduction=reduction)

def loss_calculate(y_hat, y, y_pre=None, loss_type=None, ignore_index=None, regularizer_const=None, step=None):
    loss = loss_type(y_hat, y)
    return loss

def loss_weight_balance(label):
    '''
    Qu et al., (JBHI, 2020)
    '''
    label, count = np.unique(label, return_counts=True)
    ratio_reciprocal = np.reciprocal(count/count.sum())
    loss_weight = ratio_reciprocal*(len(label)/(ratio_reciprocal.sum()))    # Weight Sum = Num of Label
    return loss_weight

''' Optimizer '''
def optimizer(params, name='network'):
    if name == 'network':
        lr = args.lr_network
    elif name == 'rss':
        lr = args.lr_rss
    opt = Adam(params, lr=lr)
    # lr_scedule = lr_scheduler.ExponentialLR(opt, gamma=0.96)
    return opt

class Optimizer():
    def __init__(self, network):
        super(Optimizer, self).__init__()
        self.opt = torch.optim.Adam(network.parameters(),
                                    lr=args.lr, weight_decay=args.wd)

    def opt_zero_grad(self):
        self.opt.zero_grad()

    def opt_step(self):
        self.opt.step()


def tensor_form(X: list, Y: list):
    for i in range(len(X)):
        X[i] = float_tensor(X[i])
    for i in range(len(Y)):
        Y[i] = long_tensor((Y[i]))
    return X, Y

class tensordataset_w_indices(Dataset[Tuple[Tensor, ...]]):
    r""" *** Custom ***
    Dataset wrapping tensors.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        # (X, Y), idx
        return tuple(tensor[index] for tensor in self.tensors), index

    def __len__(self):
        return self.tensors[0].size(0)

def dataloader_form(X_tr, Y_tr, X_val, Y_val, X_ts, Y_ts):
    tr_loader = DataLoader(tensordataset_w_indices(X_tr, Y_tr), batch_size=args.batch, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.batch, pin_memory=True)
    ts_loader = DataLoader(TensorDataset(X_ts, Y_ts), batch_size=args.batch, pin_memory=True)
    return tr_loader, val_loader, ts_loader




# Chambon: 6 EEG with 2 EOG (left and right) and 3 EMG chin channels
# Phan: the C4-A1 EEG, an average EOG (ROC-LOC), and an average EMG (CHIN1- CHIN2) channels in our experiments
# QU: 'EEG F4-LER'
# Suparatak: F4-EOG (Left)

# Channel Info
# ['EOG Left Horiz', 'EOG Right Horiz', 'EEG Fp1-LER', 'EEG Fp2-LER', 'EEG F7-LER', 'EEG F8-LER',
#  'EEG F3-LER', 'EEG F4-LER', 'EEG T3-LER', 'EEG T4-LER', 'EEG C3-LER', 'EEG C4-LER', 'EEG T5-LER',
#  'EEG T6-LER', 'EEG P3-LER', 'EEG P4-LER', 'EEG O1-LER', 'EEG O2-LER', 'EEG Fz-LER', 'EEG Cz-LER',
#  'EEG Pz-LER', 'EEG Oz-LER', 'EEG A2-LER', 'EMG Chin1', 'EMG Chin2', 'EMG Chin3', 'ECG ECGI']
# exclude 'EEG A2-LER', 'ECG ECGI', 'Resp Belt Thor', 'Resp Belt Abdo': then 20 EEG, 2 EOG, 3 EMG


