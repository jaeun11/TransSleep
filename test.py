import itertools
import numpy as np
from scipy.stats import entropy
from utils import *
from config import *
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


def predict(dataloader, network):
    Y_new, Y_hat, Y_hat_pb = np.array([]), np.array([]), np.array([[],[],[],[],[]]).reshape(0,5)
    for iteration, batch in enumerate(zip(dataloader)):
        x, y = batch[0]
        x, y = x.to(device), y.flatten().to(device)

        with torch.no_grad():
            x = network.FE(x)
            x_att, _ = network.sce(x)
            h = network.bilstm(x_att)
            x = x.flatten(start_dim=2)
            h = network.dropout(network.project_f(x) + h)
            l_2 = network.cls(h)
            l_2 = l_2.flatten(end_dim=1)

            y_hat = dcn(l_2.detach().argmax(-1))
            y_hat_pb = dcn(F.softmax(l_2, dim=-1))

        Y_new = np.concatenate([Y_new, dcn(y)])
        Y_hat = np.concatenate([Y_hat, y_hat])
        Y_hat_pb = np.concatenate([Y_hat_pb, y_hat_pb])
    return Y_hat_pb, Y_hat, Y_new