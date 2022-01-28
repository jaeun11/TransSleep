from config import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score
import numpy as np
from utils import *

def evaluate(Y_val_hat_pb, Y_val_hat, Y_val, Y_ts_hat_pb, Y_ts_hat, Y_ts, epoch, loss, log_file=None, cv_i=None):
    ''' calculate several metrics and store in cache_pf
    cache_pf: performance metrics will be stored '''
    labels = np.unique(Y_ts)
    f1_vl = round(f1_score(Y_val, Y_val_hat, labels=[0,1,2,3,4], average='macro', zero_division=1), 4)
    f1_ts = round(f1_score(Y_ts, Y_ts_hat, labels=[0,1,2,3,4], average='macro', zero_division=1), 4)
    f1_per_class = f1_score(Y_ts, Y_ts_hat, labels=[0,1,2,3,4], average=None)
    f1_per_class = {a: round(b, 4) for a, b in zip(labels, f1_per_class)}
    kappa = round(cohen_kappa_score(Y_ts, Y_ts_hat), 4)
    acc = round(accuracy_score(Y_ts, Y_ts_hat), 4)
    confusion = confusion_matrix(Y_ts, Y_ts_hat, labels=labels)
    confusion = {a: b for a, b in zip(labels, confusion)}
    ## For Lable Transition Task
    # roc_auc_val = round(roc_auc_score(Y_val, Y_val_hat_pb[:, 1]), 4)
    # roc_auc_ts = round(roc_auc_score(Y_ts, Y_ts_hat_pb[:, 1]), 4)
    # writelog(log_file, f'roc_auc_val: {roc_auc_val}, roc_auc_ts: {roc_auc_ts}')

    # For Record
    cache_pf_column = (["CV", "Epoch", " ", "Valid_F1", "Test_F1", "Kappa", "ACC", " ",
                        "Batch", "lr", "Seq_Length", " ","Training_Loss", " ",
                        "F1_per_Class", "Confusion"])
    cache_pf = [cv_i, epoch+1, " ", f1_vl, f1_ts, kappa, acc, " ",
                args.batch, args.lr, args.seq_length, " ",loss, " ",
                f1_per_class, confusion]

    return cache_pf_column, cache_pf, f1_vl, f1_ts
