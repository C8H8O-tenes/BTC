import numpy as np

import time
import math


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as DL
import torch.optim as optim
from torch import nn
from tqdm import tqdm

# from main_process import TestbedDataset, get_dict_datafile
from main_smilesDataSet import smilesDataSet
from transformer_smiles import Transformer
from transformer_smiles import Encoder
from nt_xent import NT_Xent

""" set device """
from globalvar import _init, set_value, get_value
device = get_value('cuda', torch.device('cpu'))


# train for one epoch to learn unique features
def calculate(inputs, attn_score):
    """
    统计计算attention分数最高的前三
    1. 计算attention score 最大的前三个以及索引
    2. 通过索引去找对应输入ids的位置的字符标志
    3. 通过ids去映射到字
    """
    # print(inputs.shape,len(attn_score),attn_score[0].shape)
    attn_score = torch.tensor(attn_score[0])
    scores, indexs = torch.sort(attn_score, descending=True)
    top_3_token = []
    for i in range(indexs.shape[0]):
        ids = inputs[i, indexs[i, :3]]
        # print(ids)
        top_3_token.append(ids.cpu().numpy().tolist())
    return top_3_token


def get_dict(datafile):
    # smiles 字典 统计所有smiles 字符出现频率 有高到低字典排序 1- 43
    src_dict = {}
    with open("data/pretrain/data/" + datafile + "_dict.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip()
            k = line.split(' ')[0]
            v = line.split(' ')[1]
            src_dict[k] = int(v)
    f.close()
    sort_dict = {key: rank for rank, key in enumerate(sorted(src_dict.values(), reverse=True), 1)}
    vocab_dict = {k: sort_dict[v] for k, v in src_dict.items()}

    vocab_dict['<pad>'] = 0
    return vocab_dict


def compute_rsquared(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    r2 = round((SSR / SST) ** 2, 3)
    return r2


