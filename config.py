import datetime
import os
import csv
import numpy as np
import random
import shutil
import torch
import argparse


parser = argparse.ArgumentParser(description="Experiment Info and Setings, Model Hyperparameters")
parser.add_argument("--lambda_cls", type=float, default=1)
parser.add_argument("--lambda_sc", type=float, default=2)
parser.add_argument("--lambda_st", type=float, default=0.2)
parser.add_argument("--lambda_cos_loss", type=float, default=2)
# Experiment Info
parser.add_argument("--experiment_date", type=str, default=f"{datetime.datetime.now().strftime('%Y%m%d')}")
parser.add_argument("--experiment_time", type=str, default=f"{datetime.datetime.now().strftime('%H:%M:%S')}")
parser.add_argument("--characteristic", '-c', type=str, default="")
parser.add_argument("--data", type=str, default='Sleep-edf')
parser.add_argument("--data_type", type=str, default='epoch')
parser.add_argument("--scheme", type=str, default='M_M')
parser.add_argument("--loss_weight", type=int, default=1)
parser.add_argument("--lstm_layers", type=int, default=1)
parser.add_argument("--cos_loss", type=int, default=1)
parser.add_argument("--mha", type=int, default=1)
parser.add_argument("--mha_length", type=int, default=8)
parser.add_argument("--mha_head", type=int, default=2)
parser.add_argument("--mass_ch", type=str, default='eeg_f4-ler')
parser.add_argument("--downsample", type=int, default=100)
# Experiment Hyperparameters
parser.add_argument("--epoch", type=int, default=150)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-3)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--early_stop", type=int, default=50)
parser.add_argument("--dropout", type=int, default=0.5)
parser.add_argument("--scheduler", type=int, default=0)
parser.add_argument("--stride", type=str, default=2)
parser.add_argument("--preprocess", type=str, default='robustscale')
# Model Hyperparameters
parser.add_argument("--seq_length", type=int, default=25)
# GPU
parser.add_argument("--GPU", type=bool, default=True)
parser.add_argument("--gpu_idx", type=int, default=-1)
# Experiment Sbj
parser.add_argument("--range_start", type=int, default=0)
parser.add_argument("--range_end", type=int, default=31)
args = parser.parse_args()

# Data Dependent Param
if args.data=='MASS': cv, num_sbj_total, ch, Fs = 31, 62, 0, 100
else: cv, num_sbj_total, ch, Fs = 20, 40, 0, 100

# GPU
if args.GPU:
    import GPUtil
    if args.gpu_idx == -1:
        gpu_idx = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
    else:
        gpu_idx = "%d" % args.gpu_idx
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    print(gpu_idx)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CPU Directory: /home/jep/Desktop/Pycharm/Sleep/Performance
# GPU Directory: /.../Code/Sleep/Performance



# Record File: All Contained in logger_dir
if args.GPU: log_dir = f"/.../Performance/{args.experiment_date}_{os.getcwd().split('/')[-1]}"
else: log_dir = f"/.../Performance/{args.experiment_date}_{os.getcwd().split('/')[-1]}"
if args.data=='MASS': log_dir = log_dir + '_MASS'
log_dir = log_dir + f"/{args.experiment_time}_{args.characteristic}"
print(log_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)

# CSV File
log_csv_file = log_dir + "/" + args.experiment_date + "_" + args.characteristic + ".csv"
log_csv_file_val = log_dir + "/" + args.experiment_date + "_" + args.characteristic + "_val.csv"
# if file not exists, create, else: append: experiment class iterate per sbj and cv
if not os.path.isfile(log_csv_file):
    with open(log_csv_file, 'w', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Code directory", os.getcwd().split('/')[-1]])

# Save python Code
files_tmp, files = os.listdir(os.getcwd()), []
if os.path.isdir(log_dir + '/code'): pass
else: os.mkdir(log_dir + '/code')
# If Exists, Overwrite
for f in files_tmp:
    if f.split('.')[-1]=='py': files.append(f)
for file in files: shutil.copy(os.getcwd() + '/' + file, log_dir + '/code/' + file)  # save python code

# Seed
def set_seed(seed = 961125):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



# Data Info
if args.data == 'Sleep-edf':
    data_path = "/.../Dataset/Sleep/Sleep-edf" if not args.GPU else "/.../Dataset/Sleep-edf"
    X_path_dict = {'embed': "/Sleep_Embedded", 'AE': "/Sleep_AE_per_modality_0.5HZ",
                  'raw': '/Sleep_Raw', 'epoch':'/Sleep_Epoch'}
    Y_path_dict = {'raw': '/Sleep_Raw', 'epoch': '/Sleep_Epoch'}
if args.data == 'MASS':
    data_path = "/.../Dataset/Sleep/MASS/SS3" if not args.GPU else "/.../Dataset/MASS/SS3"
    X_path_dict = {'epoch': '/SS3_Epoch'}
    Y_path_dict = {'epoch': '/SS3_Epoch'}



