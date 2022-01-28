from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import robust_scale
from mne.filter import filter_data
from utils import *

def robustscaler(x_tr, x_val, x_ts):
    scaler = RobustScaler()
    x_tr_sh = x_tr.shape
    x_val_sh = x_val.shape
    x_ts_sh = x_ts.shape
    x_tr = scaler.fit_transform(x_tr.reshape(-1, x_tr_sh[-1]))
    x_val = scaler.transform(x_val.reshape(-1, x_val_sh[-1]))
    x_ts = scaler.transform(x_ts.reshape(-1, x_ts_sh[-1]))
    x_tr = x_tr.reshape(*x_tr_sh)
    x_val = x_val.reshape(*x_val_sh)
    x_ts = x_ts.reshape(*x_ts_sh)
    return x_tr, x_val, x_ts

def bandpass(x, sf=Fs):
    x = flatten_1dim(x)
    x = filter_data(x, sfreq=sf, l_freq=0.5, h_freq=49.9)
    x = x.reshape(-1, 30*sf)
    return x
