import os
import datetime
import csv
import sys
import numpy as np
import tqdm
import model
import utils
from utils import *
from config import *
from preprocess import *
from data import *
from model import *
import train, test
import evaluate
import torch


def main(tr_loader, val_loader, ts_loader, cv_i, log_file, loss_weight=None, Y_t=None):
    # Initialize Network
    network = model.Model(x=tr_loader.dataset.tensors[0][:2]).to(device)

    # Initialize Optimizer, Scheduler
    opt = utils.Optimizer(network)

    patience, pf_callback = 0, 0
    for epoch in tqdm.trange(args.epoch, desc=record_name(cv_i)):
        if patience < args.early_stop:
            # Train
            network.train()
            loss_network, f1_tr = train.trainer(tr_loader, network=network, opt=opt,
                                                loss_weight=loss_weight, Y_t=Y_t['tr'], log_file=log_file)

            # Test
            network.eval()
            Y_val_hat_pb, Y_val_hat, Y_val = test.predict(val_loader, network)
            Y_ts_hat_pb, Y_ts_hat, Y_ts = test.predict(ts_loader, network)
            Y_val, Y_ts = dcn(flatten_1dim(Y_val)), dcn(flatten_1dim(Y_ts))

            # Evaluate
            pf_metric_column, pf_metric, f1_val, f1_ts = \
                evaluate.evaluate(Y_val_hat_pb, Y_val_hat, Y_val, Y_ts_hat_pb, Y_ts_hat, Y_ts, epoch, loss_network,
                                  log_file=log_file, cv_i=cv_i)
            # Early Stop, Record Update
            pf_val = f1_val
            if float(pf_callback) < float(pf_val): patience, pf_callback = 0, float(pf_val)    # Reset
            else: patience += 1
        else:
            break


# Data Load
X_dict, Y_dict, idx_sbj_total = Data_Loader()    # [B, F, T]
for cv_i in range(args.range_start, args.range_end):
    set_seed()
    # log File
    log_file = log_dir + "/" + f"performance_per_epoch_SBJ_{cv_i}.log"
    if not os.path.isfile(log_file):
        with open(log_file, 'w', encoding='utf-8') as f: pass
    writelog(log_file, f'{sys.argv}\n')    # Save Command Line Arguments
    writelog(log_file, f'data: {args.data}')
    writelog(log_file, f'Sbj_{cv_i:02} X_dict keys Num): {len(X_dict.keys())}\n')    # Num of Data

    # Prepare Train Validation Test Dataset
    if args.data == 'Sleep-edf': idx_sbj_ts = [i for i in list(X_dict.keys()) if i.startswith(f'{cv_i:02}')]
    if args.data == 'MASS': idx_sbj_ts = list(X_dict.keys())[cv_i * 2:(cv_i + 1) * 2]
    print("Starting Training SBJ {}".format(idx_sbj_ts))

    # Split Data
    X_tr, Y_tr_org, X_val, Y_val_org, X_ts, Y_ts_org = tr_val_ts_split(idx_sbj_ts, idx_sbj_total, X_dict, Y_dict, log_file)
    if args.preprocess=='robustscale':
        writelog(log_file, f'preprocessing, {args.preprocess}')
        X_tr, X_val, X_ts = robustscaler(X_tr, X_val, X_ts)

    # log Data Info
    label, count = np.unique(Y_tr_org, return_counts=True)
    [X_tr, X_val, X_ts], [Y_tr_org, Y_val_org, Y_ts_org] = tensor_form([X_tr, X_val, X_ts], [Y_tr_org, Y_val_org, Y_ts_org])
    writelog(log_file, f'X_tr Shape: {X_tr.shape}, X_val Shape: {X_val.shape}, X_ts Shape: {X_ts.shape}\n'
                       f'Training Original Label Count: {dict(zip(label, count))}')

    # Labeling
    Y_tr_t, Y_val_t, Y_ts_t = label_stage_transition([Y_tr_org, Y_val_org, Y_ts_org], log_file)
    label, count = np.unique(Y_tr_t, return_counts=True)
    [], [Y_tr_t, Y_val_t, Y_ts_t] = tensor_form([], [Y_tr_t, Y_val_t, Y_ts_t])
    writelog(log_file, f'Training Transition Label Count: {dict(zip(label, count))}')
    # ***** Transition Label Becomes Auxiliary Label *****
    Y_t = {'tr': Y_tr_t, 'val': Y_val_t, 'ts': Y_ts_t}

    # Balance Loss Weight
    if args.loss_weight:
        loss_weight_org = float_tensor(loss_weight_balance(Y_tr_org)).to(device)    # tensor array
        loss_weight_t = float_tensor(loss_weight_balance(Y_tr_t)).to(device)  # tensor array
        loss_weight_dict = {'org':loss_weight_org, 'trans':loss_weight_t}
    else: loss_weight = None
    writelog(log_file, f'Balanced Loss Weight: {loss_weight_dict}')
    writelog(log_file, f'Balanced Loss Weight Sum: {loss_weight_org.sum()}')
    tr_loader, val_loader, ts_loader = dataloader_form(X_tr, Y_tr_org, X_val, Y_val_org, X_ts, Y_ts_org)
    main(tr_loader, val_loader, ts_loader, cv_i, log_file, loss_weight_dict, Y_t=Y_t)

