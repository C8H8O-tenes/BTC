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
                       default='/mnt/pan/xuhaixia/bystander_model_03/predict.yaml')
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
device = config['Train']['Model_Parameter']['device']
patience = config['Train']['Trainer_parameter']['patience']

# data_path = "/mnt/pan/xuhaixia/preprocess/GSE179994_RAW.h5ad"
data_path = "/mnt/pan/xuhaixia/preprocess/finetune_blood.h5ad"
adata = sc.read_h5ad(data_path)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000)
adata = adata[:, adata.var['highly_variable']]
sc.pp.scale(adata)

# adata.obs["CellType"] = adata.obs.cluster
adata.obs["CellType"] = adata.obs["sample"]
dataset = PretrainDataset(adata)
smile_seqs, vocab_dict = dataset.get_smile_seqs()

print("over")


# initialize a new model
TRB_TC_example = TRB_TC(
    d_model=config['Train']['Model_Parameter']['d_model'],
    d_ff=config['Train']['Model_Parameter']['d_ff'], d_k=config['Train']['Model_Parameter']['d_k'], d_v=config['Train']['Model_Parameter']['d_v'],
    n_heads=config['Train']['Model_Parameter']['n_heads'], n_layers=config['Train']['Model_Parameter']['n_layers'],
    precet=config['Train']['Model_Parameter']['precet'],
    seq_len=smile_seqs[0].shape[0],

    in_dim=config['Train']['Model_Parameter']['encoderprofile_in_dim'],
    hid_dim=config['Train']['Model_Parameter']['encoderprofile_hid_dim'],
    hid_dim2=config['Train']['Model_Parameter']['encoderprofile_hid_dim2'],
    out_dim=config['Train']['Model_Parameter']['encoderprofile_out_dim'],
    dropout=0.2,
    temperature=0.1
    )
TRB_TC_example = TRB_TC_example.to(device)

# epoch = 40
# checkpoint_path = f"{config['Train']['output_dir']}/Model_checkpoints/merge_finetune_epoch{epoch}_model.pth"
checkpoint_path = f"{config['Train']['output_dir']}/Model_checkpoints/3d_blood_best_model.pth"
print("checkpoint_path:", checkpoint_path)

checkpoint = torch.load(checkpoint_path)
TRB_TC_example.load_state_dict(checkpoint['model_state_dict'])
TRB_TC_example.eval()



# initialize the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['Train']['Sampling']['batch_size'],
                                          shuffle=False)
# initialize the logger for saving the traininig log
logger = get_logger(config['Train']['output_dir']+'/training.log')

# # setting the training epoch
# epochs = config['Train']['Trainer_parameter']['epoch']

all_label = []
with torch.no_grad():
    TCR_embeddings = []
    Profile_embeddings = []
    Gaps = []
    s = time.time()

    train_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, tem in train_bar:
        TCR_beta, batch_genematrix, TCR_beta_dict, index, GEO_label, sample_label, CellType_label = tem

        encoderprofile_embedding, encoderTCR_embedding, pred_label = TRB_TC_example.predict_forward(
            [TCR_beta.to(device), batch_genematrix.to(device)], # x
            task_level="predict")

        prob = torch.sigmoid(pred_label)
        predicted_classes = (prob >= 0.5).long()
        all_label.append(predicted_classes)

        Profile_embeddings.append(encoderprofile_embedding.cpu())
        TCR_embeddings.append(encoderTCR_embedding.cpu())

        tmp_gap = 2 * (1 - (encoderTCR_embedding @ encoderprofile_embedding.T).diag())
        Gaps.append(tmp_gap.cpu())
    e = time.time()

print("time", e-s)

# reduce
TCR_vector = torch.cat(TCR_embeddings, dim=0)
Profile_vector = torch.cat(Profile_embeddings, dim=0)
Gap_vector = torch.cat(Gaps, dim=0)
predicet_labels = torch.cat(all_label, dim=0).to('cpu')

predicet_labels_list = [label.item() for label in predicet_labels]
print(Counter(predicet_labels_list))
label_mapping = torch.load("/mnt/pan/xuhaixia/preprocess/label_mapping.pt")
predicted_labels = [key for label in predicet_labels_list for key, value in label_mapping.items() if value == label]

logger.info('finish predicting!')


combined_features = torch.cat((Profile_vector, TCR_vector), dim=1)
umap_embedding = umap.UMAP(n_neighbors=80, min_dist=0.1, metric='euclidean').fit_transform(combined_features.detach().numpy())
gene_umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(Profile_vector.detach().numpy())
tcr_umap_embedding = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean').fit_transform(TCR_vector.to('cpu').detach().numpy())

data = dataset.adata
data.obs.beta = data.obs.beta.astype('category')
data.obs.celltype = data.obs.celltype.astype('category') # celltype
data.obs.cluster = data.obs.cluster.astype('category') # cluster
data.obs.response = data.obs.response.astype('category') # cluster

data.obs['label'] = predicted_labels
data.obs['label'] = data.obs['label'].astype('category') # label

data.obsm['X_umap'] = umap_embedding
data.obsm['gene_umap'] = gene_umap_embedding
data.obsm['tcr_umap'] = tcr_umap_embedding

# save
torch.save(Profile_vector, '/mnt/pan/xuhaixia/bystander_model_03/TrainingResult/Model_checkpoints/GSE179994_profile_vector.pt')
torch.save(TCR_vector, '/mnt/pan/xuhaixia/bystander_model_03/TrainingResult/Model_checkpoints/GSE179994_tcr_vector.pt')
data.raw.var.columns = ["Gene"]
data.write_h5ad("/mnt/pan/xuhaixia/bystander_model_03/TrainingResult/GSE179994_reduce_data.h5ad")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
plt.figure(figsize=(6, 5))
cmap = plt.get_cmap('Set3')  # 或者尝试 'Set3', 'tab10', 'Dark2' 等离散调色板
norm = mcolors.BoundaryNorm(range(len(set(data.obs["label"].cat.codes)) + 1), cmap.N)
plt.scatter(data.obsm['X_umap'][:, 0],
			data.obsm['X_umap'][:, 1],
            c=data.obs["label"].cat.codes, cmap=cmap, s=0.5, norm=norm)
plt.title('GSE179994')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
legend_labels = list(set(data.obs["label"]))
legend_labels = [str(label) for label in legend_labels]
handles = [mpatches.Patch(color=cmap(i / len(legend_labels)), label=label) for i, label in enumerate(legend_labels)]
plt.legend(handles=handles, title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.savefig("/mnt/pan/xuhaixia/bystander_model_03/TrainingResult/GSE179994_allumap_reduce.png", dpi=300)
plt.show()

#### gene
# gene_pred
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap('tab20')  # 或者尝试 'Set3', 'tab10', 'Dark2' 等离散调色板
norm = mcolors.BoundaryNorm(range(len(set(data.obs["label"].cat.codes)) + 1), cmap.N)
plt.scatter(gene_umap_embedding[:, 0], gene_umap_embedding[:, 1],
            c=data.obs["label"].cat.codes, cmap=cmap, s=0.5, norm=norm)
plt.title('GSE179994')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
legend_labels = list(set(data.obs["label"]))
legend_labels = [str(label) for label in legend_labels]
handles = [mpatches.Patch(color=cmap(i / len(legend_labels)), label=label) for i, label in enumerate(legend_labels)]
plt.legend(handles=handles, title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.savefig("/mnt/pan/xuhaixia/bystander_model_03/TrainingResult/GSE179994_gene_umap_reduce.png", dpi=300)
plt.show()

##### tcr
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap('tab20')  # 或者尝试 'Set3', 'tab10', 'Dark2' 等离散调色板
norm = mcolors.BoundaryNorm(range(len(set(data.obs["label"].cat.codes)) + 1), cmap.N)
plt.scatter(tcr_umap_embedding[:, 0], tcr_umap_embedding[:, 1],
            c=data.obs["label"].cat.codes, cmap=cmap, s=0.5, norm=norm)
plt.title('GSE179994')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
legend_labels = list(set(data.obs["label"]))
legend_labels = [str(label) for label in legend_labels]
handles = [mpatches.Patch(color=cmap(i / len(legend_labels)), label=label) for i, label in enumerate(legend_labels)]
plt.legend(handles=handles, title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.savefig("/mnt/pan/xuhaixia/bystander_model_03/TrainingResult/GSE179994_tcr_reduce.png", dpi=300)  #
plt.show()
print("over")


# to seurat
import h5py
meta = pd.DataFrame(data=data.obs)
meta.to_csv('/mnt/pan/xuhaixia/bystander_model_03/TrainingResult/GSE179994_pred_metadata.csv')

mat = pd.DataFrame(data=data.X, index=data.obs_names, columns=data.var_names)
with h5py.File("/mnt/pan/xuhaixia/bystander_model_03/TrainingResult/GSE179994_pred.h5", "w") as f:
    f.create_dataset("mat", data=mat.values)  # 保存矩阵数据
    f.create_dataset("obs_names", data=mat.index.values)  # 细胞名称
    f.create_dataset("var_names", data=mat.columns.values)  # 基因名称
    f.create_dataset("X_umap", data=data.obsm['X_umap'])  # X_umap 数据
    f.create_dataset("gene_umap", data=data.obsm['gene_umap'])  # gene_umap 数据
    f.create_dataset("tcr_umap", data=data.obsm['tcr_umap'])  # tcr_umap 数据

