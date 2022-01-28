import csv
import random
import glob2
from config import *
from utils import *
from preprocess import *
import numpy as np
import torch
# from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data.sampler import WeightedRandomSampler



'''Make Sliding Window'''
def X_window_maker(x, scheme):
    if scheme=='M_M':
        remain = x.shape[0] % args.seq_length    # Sliding Window and Permute
        x_window = x[remain:].reshape(-1, args.seq_length, x.shape[-1])    # Slice The Remained From Front
        return x_window
    elif scheme=='O_O': return x

def Y_window_maker(y, scheme):
    if scheme=='M_O' or scheme=='O_O': y_window = y
    if scheme=='M_M':
        remain = y.shape[0] % args.seq_length
        y_window = y[remain:].reshape(-1, args.seq_length)
    return y_window


'''
Data Loader: Should Exploit Sliding Window before Spliting Train/Val 
'''
def Data_Loader():
    '''
    Load Total Data in a Dict
    Key: Idx_Sbj
    Value: data
    '''
    # Data file
    X_path = data_path + X_path_dict[args.data_type]
    Y_path = data_path + Y_path_dict[args.data_type]
    X_file_list = sorted(glob2.glob(X_path + '/data*.npy'))
    if args.data == 'MASS':
        if args.mass_ch == 'eeg_f4-ler': X_file_list = sorted(glob2.glob(X_path + '/EEG_F4-LER_*.npy'))
        if args.mass_ch == 'f4-eog': X_file_list = sorted(glob2.glob(X_path + '/F4-EOG_*.npy'))
    Y_file_list = sorted(glob2.glob(Y_path + '/label*.npy'))

    # Total Sbj Idx by File Name
    sbj_total_list = []
    if args.data == 'MASS':    # 43, 49 doesn't exist
        for i in range(len(X_file_list)): sbj_total_list.append(X_file_list[i].split('.')[0][-2:])
    if args.data == 'Sleep-edf':
        X_file_list, Y_file_list = X_file_list[:39], Y_file_list[:39]
        for i in range(len(X_file_list)): sbj_total_list.append(X_file_list[i].split('.')[0][-4:])    # Sbj_Night

    # Sliding Window per Sample
    if args.data == 'MASS':
        print(f'args.downsample: {args.downsample}')
        if args.downsample == 100: X_list = [
            X_window_maker(bandpass(downsample_to_100(np.load(i, mmap_mode='r'))), args.scheme) for i in X_file_list]
    if args.data == 'Sleep-edf':
        X_list = [X_window_maker(bandpass(np.load(i, mmap_mode='r')[:, ch]), args.scheme) for i in X_file_list]

    Y_list = [Y_window_maker(np.load(i), args.scheme) for i in Y_file_list]
    X_dict = dict(zip(sbj_total_list, X_list))
    Y_dict = dict(zip(sbj_total_list, Y_list))
    return X_dict, Y_dict, sbj_total_list

# data_loader = Data_Loader()
# X_dict, Y_dict, sbj_total_list = data_loader()
# print(1)
