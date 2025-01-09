import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import scanpy as sc
import joblib
import re
import time


class FinetuneDataset(Dataset):
    def __init__(self, adata):
        super(FinetuneDataset, self).__init__()
        self.adata = adata
        self.genematrix = self.adata.X.astype(np.float32)

        self.tcrcdr3 = self.adata.obs['beta'] # TCRCDR3
        self.target_length = 25

        self.smile_seqs, self.vocab_dict = self.tcrdis_process()
        self.TCR_ids = {beta: idx for idx, beta in enumerate(self.tcrcdr3.unique())}

        self.label, self.label_mapping, _ = self.read_label()


    def tcrdis_process(self):
        def get_dict(smile):
            smile_list = []
            for s in smile:
                smile_list.append(s)
            return smile_list

        # Count the number of character occurrences in each sequence and build a vocabulary dictionary
        org_dict = {} # 统计每个氨基酸字母出现的频数
        data_list = []
        for smile in self.tcrcdr3:
            smiles = get_dict(smile)
            data_list.append(smiles)
            for s in smiles:
                org_dict[s] = org_dict.get(s, 0) + 1

        # add <pad>
        vocab_dict = {k: i + 1 for i, (k, _) in enumerate(sorted(org_dict.items(), key=lambda x: x[1], reverse=True))}
        vocab_dict['<pad>'] = 0 # # 按照频数将氨基酸字母转为数字：{'S': 1, 'F': 2……, '<pad>': 0}
        # vl = len(vocab_dict)
        PAD_IDX = vocab_dict['<pad>']

        # Convert the sequence to a numeric sequence and fill it to the target length
        # 将序列转换为数值序列，并填充到目标长度
        smile_seqs = [
            F.pad(torch.LongTensor([vocab_dict.get(i, PAD_IDX) for i in get_dict(smile)]),
                  (0, self.target_length - len(smile)),
                  value=PAD_IDX)
            for smile in self.tcrcdr3
        ]

        return smile_seqs, vocab_dict

    def read_label(self):
        label_fs = self.adata.obs.label
        label_fs_categories = pd.Categorical(label_fs)
        num_categories = len(label_fs.unique())
        print(f"num_categories:{num_categories}")

        label_mapping = {category: code for code, category in enumerate(label_fs_categories.categories)}
        print(label_mapping)

        # label tp int
        label_fs_codes = label_fs_categories.codes
        label_fs_codes = np.array(label_fs_codes).astype('int32')
        label_fs_tensor = torch.from_numpy(label_fs_codes)
        label_fs_tensor = label_fs_tensor.type(torch.LongTensor)

        return label_fs_tensor, label_mapping, num_categories

    def get_smile_seqs(self):
        return self.smile_seqs, self.vocab_dict
    def get_label(self):
        return self.label,self.label_mapping

    def __getitem__(self, index):
        return (self.smile_seqs[index],
                self.genematrix[index],
                self.TCR_ids[self.tcrcdr3.iloc[index]],
                self.label[index],
                index
                )

    def __len__(self):
        return len(self.smile_seqs)
