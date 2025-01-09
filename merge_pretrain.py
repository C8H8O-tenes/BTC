import scanpy as sc
import pandas as pd
import numpy as np
import random
import time
from collections import Counter

import torch
import torch.nn.init as init

from torch.utils.data import DataLoader
from main_tcr_train import *
from merge_Dataset import PretrainDataset
import scipy as sp

import pickle
import umap
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

import logging
import yaml
import argparse

from utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, help='the path to the config file (*.yaml)',
                       default='/mnt/pan/xuhaixia/bystander_model_03/test.yaml')
args = argparser.parse_args()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# load the cofig file
config_file = args.config
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file)

# random seed setting
torch.manual_seed(config['Train']['Trainer_parameter']['random_seed'])
torch.cuda.manual_seed_all(config['Train']['Trainer_parameter']['random_seed'])
np.random.seed(config['Train']['Trainer_parameter']['random_seed'])
random.seed(config['Train']['Trainer_parameter']['random_seed'])
torch.cuda.manual_seed(config['Train']['Trainer_parameter']['random_seed'])

# device = config['Train']['Model_Parameter']['device']

device_id = 2  # 选择第 0 个 GPU，修改此值可以选择其他 GPU
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

patience = config['Train']['Trainer_parameter']['patience']

# pretrein_path = "/mnt/pan/xuhaixia/preprocess/pretrain_data.h5ad"
# pretrein_path = "/mnt/pan/xuhaixia/preprocess/subpretrain_data.h5ad"
# pretrein_path = "/mnt/pan/xuhaixia/preprocess/bystander_result.h5ad"
# adata = adata[adata.obs["sample"].str.contains('tumor'), :]
# adata.raw.var.columns=['Gene']
# adata.write("/mnt/pan/xuhaixia/preprocess/bystander_result_tumor.h5ad")
pretrein_path = "/mnt/pan/xuhaixia/preprocess/merge_pretraindata_8datasets_afterscanpy.h5ad"
adata = sc.read_h5ad(pretrein_path)
# adata = adata[~(adata.X.sum(axis=1) == 0), :]
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata, n_top_genes=5000)
# adata = adata[:, adata.var['highly_variable']]
# sc.pp.scale(adata)
#
# adata.obs["CellType"] = adata.obs["sample"]
# # adata.obs["CellType"] = adata.obs["CellType"].str.replace(r'\(.*\)', '', regex=True)
dataset = PretrainDataset(adata)
smile_seqs, vocab_dict = dataset.get_smile_seqs()

print("over")


# initialize a new model
TRB_TC_example = TRB_TC(
    d_model=32,
    d_ff=128, d_k=128, d_v=128,
    n_heads=6, n_layers=6,
    precet=0.25,
    seq_len=smile_seqs[0].shape[0],

    in_dim=5000,
    hid_dim=1024,
    hid_dim2=512,
    out_dim=128,
    dropout=0.2,
    temperature=0.1
    )
TRB_TC_example = TRB_TC_example.to(device)


# setting the learning rate for the model
TRB_TC_example.optimizer = torch.optim.AdamW(TRB_TC_example.parameters(), lr = config['Train']['Trainer_parameter']['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(TRB_TC_example.optimizer, mode='min', factor=0.5, patience=10)
# initialize the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['Train']['Sampling']['batch_size'],
                                          shuffle=config['Train']['Sampling']['sample_shuffle'])
# initialize the logger for saving the traininig log
logger = get_logger(config['Train']['output_dir']+'/training.log')

# setting the training epoch
epochs = config['Train']['Trainer_parameter']['epoch']

# training
best_loss = float('inf')
counter = 0

all_loss = []
GEO_all_nmi_scores = []
sample_all_nmi_scores = []
CellType_all_nmi_scores = []
number_of_GEO = len(Counter(dataset.adata.obs.GEO))
number_of_sample = len(Counter(dataset.adata.obs["sample"]))
number_of_CellType = len(Counter(dataset.adata.obs["CellType"]))

def compute_nmi_loss(embeddings, labels, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0).fit(embeddings)
    cluster_labels = kmeans.labels_
    nmi_score = normalized_mutual_info_score(labels, cluster_labels)  # 计算 NMI 分数
    return 1 - nmi_score

for epoch in range(1, epochs + 1):
    recon_loss_epoch, loss_epoch = 0, 0
    all_embeddings, all_GEO_labels, all_sample_labels, all_CellType_labels = [], [], [], []
    s = time.time()
    nmi_loss_weight = 0.05

    train_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, tem in train_bar:
        TCR_beta, batch_genematrix, TCR_beta_dict, index, GEO_label, sample_label, CellType_label = tem

        Contrastive_loss, Recon_loss, attn_score, encoderprofile_embedding, encoderTCR_embedding = TRB_TC_example.pretrain_forward(
            [TCR_beta.to(device), batch_genematrix.to(device)], # x
            task_level="pretrain"
            )

        # 获取梯度
        gradients = torch.autograd.grad(Contrastive_loss, TRB_TC_example.parameters(), retain_graph=True,
                                        allow_unused=True)
        contrastive_grad = torch.norm(torch.cat([grad.view(-1) for grad in gradients if grad is not None]), p=2)
        gradients = torch.autograd.grad(Recon_loss, TRB_TC_example.parameters(), retain_graph=True, allow_unused=True)
        recon_grad = torch.norm(torch.cat([grad.view(-1) for grad in gradients if grad is not None]), p=2)
        # 计算梯度大小的比率来调整权重
        total_grad = contrastive_grad + recon_grad
        contrastive_loss_weight = contrastive_grad / total_grad
        recon_loss_weight = recon_grad / total_grad

        embeddings = np.concatenate((encoderTCR_embedding.detach().cpu().numpy(), encoderprofile_embedding.detach().cpu().numpy()), axis=1)
        celltype_nmi_loss = compute_nmi_loss(embeddings, CellType_label, number_of_CellType)

        # 计算最终损失
        loss_epoch = (
                contrastive_loss_weight * Contrastive_loss +
                recon_loss_weight * Recon_loss +
                nmi_loss_weight * celltype_nmi_loss  # 控制 NMI 损失的权重
        )
        TRB_TC_example.optimizer.zero_grad()
        loss_epoch.backward()
        TRB_TC_example.optimizer.step()

        encoderTCR_embedding_np = encoderTCR_embedding.detach().cpu().numpy()
        encoderprofile_embedding_np = encoderprofile_embedding.detach().cpu().numpy()
        embeddings = np.concatenate((encoderTCR_embedding_np, encoderprofile_embedding_np), axis=1)
        all_embeddings.append(embeddings)
        all_GEO_labels.extend(GEO_label)
        all_sample_labels.extend(sample_label)
        all_CellType_labels.extend(CellType_label)

    all_loss.append(loss_epoch)
    scheduler.step(loss_epoch)

    e = time.time()

    # 聚类指标
    label_encoder = LabelEncoder()
    GEO_labels_encoded = label_encoder.fit_transform(all_GEO_labels)
    sample_labels_encoded = label_encoder.fit_transform(all_sample_labels)
    CellType_labels_encoded = label_encoder.fit_transform(all_CellType_labels)
    # 将 all_embeddings 转为 NumPy 数组
    all_embeddings_np = np.vstack(all_embeddings)

    GEO_kmeans = KMeans(n_clusters=number_of_GEO, n_init=20, random_state=0).fit(all_embeddings_np)
    GEO_cluster_labels = GEO_kmeans.labels_
    sample_kmeans = KMeans(n_clusters=number_of_sample, n_init=20, random_state=0).fit(all_embeddings_np)
    sample_cluster_labels = sample_kmeans.labels_
    CellType_kmeans = KMeans(n_clusters=number_of_CellType, n_init=20, random_state=0).fit(all_embeddings_np)
    CellType_cluster_labels = CellType_kmeans.labels_
    # 计算 NMI
    GEO_nmi_score = normalized_mutual_info_score(GEO_labels_encoded, GEO_cluster_labels)
    GEO_all_nmi_scores.append(GEO_nmi_score)
    sample_nmi_score = normalized_mutual_info_score(sample_labels_encoded, sample_cluster_labels)
    sample_all_nmi_scores.append(sample_nmi_score)
    CellType_nmi_score = normalized_mutual_info_score(CellType_labels_encoded, CellType_cluster_labels)
    CellType_all_nmi_scores.append(CellType_nmi_score)

    logger.info(f"GEO NMI: {GEO_nmi_score:.5f},sample NMI: {sample_nmi_score:.5f},CellType NMI: {CellType_nmi_score:.5f}")
    logger.info('Epoch:[{}/{}]\nloss_epoch:{:.5f}\tRecon_loss:{:.5f}\tContrastive_loss:{:.5f}\ttime:{:.3f}'.format(epoch, epochs, loss_epoch, recon_loss_weight * Recon_loss, contrastive_loss_weight * Contrastive_loss, e-s))

    if loss_epoch < best_loss:
        best_loss = loss_epoch
        counter = 0

        torch.save({'epoch': epoch,
                    'model_state_dict': TRB_TC_example.state_dict(),
                    'optimizer_state_dict': TRB_TC_example.optimizer.state_dict(),},
                   f"{config['Train']['output_dir']}/Model_checkpoints/merge_pretrain_epoch{epoch}_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping at epoch {epoch} as no improvement was seen for {patience} consecutive epochs.")
        break

logger.info('finish training!')

all_loss = [loss.item() for loss in all_loss]
plt.figure(figsize=(10, 6))
plt.plot(all_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()

# 聚类指标
nmi_scores_all = [loss.item() for loss in sample_all_nmi_scores]
plt.figure(figsize=(10, 6))
plt.plot(nmi_scores_all, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('nmi_scores')
plt.legend()
plt.grid(True)
plt.show()

print("over")


# 聚类可视化
import matplotlib.pyplot as plt
import numpy as np
import umap

# 使用 UMAP 降维
reducer = umap.UMAP(n_components=2, random_state=0)
reduced_embeddings_umap = reducer.fit_transform(all_embeddings_np)

# 设置类别颜色
unique_labels = np.unique(CellType_cluster_labels)
num_clusters = len(unique_labels)
colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))  # 使用离散颜色映射

# 绘制图像
plt.figure(figsize=(8, 6))
for i, label in enumerate(unique_labels):
    plt.scatter(
        reduced_embeddings_umap[CellType_cluster_labels == label, 0],
        reduced_embeddings_umap[CellType_cluster_labels == label, 1],
        color=colors[i], label=label, s=1
    )
plt.legend(title="GEO Labels", loc='upper right', markerscale=5, fontsize=10, title_fontsize=12)
plt.title('UMAP Visualization of GEO Clusters', fontsize=14)
plt.xlabel('UMAP Component 1', fontsize=12)
plt.ylabel('UMAP Component 2', fontsize=12)
plt.show()
