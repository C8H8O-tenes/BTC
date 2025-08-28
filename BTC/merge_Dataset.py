import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial.distance import euclidean
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import scanpy as sc
import joblib
import re
import time
from collections import Counter

class PretrainDataset(Dataset):
    def __init__(self, adata, target_length=25):
        """
        Dataset for pretraining using TCR sequences, gene expression, and metadata (GEO, type).
        """
        self.adata = adata
        if sp.issparse(adata.X):
            self.genematrix = np.array(adata.X.toarray(), dtype=np.float32)
        else:
            self.genematrix = np.array(adata.X, dtype=np.float32)
        self.tcrcdr3 = adata.obs['cdr3']
        self.target_length = target_length

        self.smile_seqs, self.vocab_dict = self._encode_tcr()
        self.TCR_ids = {seq: idx for idx, seq in enumerate(self.tcrcdr3.unique())}

        self.GEOlabel = adata.obs['orig.ident']
        self.type = adata.obs['celltypist_labels']
        self.cdr3 = adata.obs['cdr3']

    def _encode_tcr(self):
        """
        Tokenize and pad CDR3 sequences into fixed-length tensors.
        """
        vocab = {}
        sequences = []

        for seq in self.tcrcdr3:
            chars = list(seq)
            sequences.append(chars)
            for c in chars:
                vocab[c] = vocab.get(c, 0) + 1

        vocab_dict = {c: i + 1 for i, (c, _) in enumerate(sorted(vocab.items(), key=lambda x: x[1], reverse=True))}
        vocab_dict['<pad>'] = 0
        pad_idx = vocab_dict['<pad>']

        padded_seqs = [
            F.pad(torch.tensor([vocab_dict.get(c, pad_idx) for c in chars], dtype=torch.long),
                  (0, self.target_length - len(chars)),
                  value=pad_idx)
            for chars in sequences
        ]

        return padded_seqs, vocab_dict

    def __getitem__(self, index):
        """
        Return a single training sample including:
        - Padded TCR sequence
        - Gene expression vector
        - TCR ID
        - Cell index
        - GEO label
        - type label
        """
        return (
            # TCR_beta, gene_matrix, TCR_ids, index, GEO_label, type_label, beta_label
            self.smile_seqs[index], 
            self.genematrix[index],
            self.TCR_ids[self.tcrcdr3.iloc[index]],
            index,
            self.GEOlabel.iloc[index],
            self.type.iloc[index],
            self.cdr3.iloc[index]
        )

    def __len__(self):
        return len(self.smile_seqs)

    def get_smile_seqs(self):
        return self.smile_seqs, self.vocab_dict

class FinetuneDataset(Dataset):
    def __init__(self, adata, target_length=25, vocab_dict=None):
        """
        Custom Dataset for fine-tuning with AnnData input.
        Includes gene expression, TCR sequences, and cell-level labels.
        """
        self.adata = adata
        if sp.issparse(adata.X):
            self.genematrix = np.array(adata.X.toarray(), dtype=np.float32)
        else:
            self.genematrix = np.array(adata.X, dtype=np.float32)
        self.tcrcdr3 = adata.obs['cdr3']  # TCR cdr3-chain CDR3 sequences
        self.target_length = target_length  # Padding length for TCR sequences

        if vocab_dict is None:
            self.smile_seqs, self.vocab_dict = self._encode_tcr()
        else:
            self.vocab_dict = vocab_dict
            self.smile_seqs = self._encode_with_vocab(vocab_dict)

        # Map each unique TCR sequence to a unique ID
        self.TCR_ids = {seq: idx for idx, seq in enumerate(self.tcrcdr3.unique())}

        # Convert labels to integer class codes
        self.label, self.label_mapping = self._encode_labels()

    def _encode_with_vocab(self, vocab_dict):
        pad_idx = vocab_dict['<pad>']
        sequences = [list(seq) for seq in self.tcrcdr3]
        return [
            F.pad(torch.tensor([vocab_dict.get(c, pad_idx) for c in chars], dtype=torch.long),
                  (0, self.target_length - len(chars)),
                  value=pad_idx)
            for chars in sequences
        ]

    def _encode_tcr(self):
        """
        Tokenize and pad TCR cdr3 sequences to fixed length.
        Also build a character vocabulary based on frequency.
        """
        vocab = {}
        sequences = []

        # Build character frequency dictionary and character lists
        for seq in self.tcrcdr3:
            chars = list(seq)
            sequences.append(chars)
            for c in chars:
                vocab[c] = vocab.get(c, 0) + 1

        # Create vocabulary dict sorted by frequency, with <pad> as 0
        vocab_dict = {c: i + 1 for i, (c, _) in enumerate(sorted(vocab.items(), key=lambda x: x[1], reverse=True))}
        vocab_dict['<pad>'] = 0
        pad_idx = vocab_dict['<pad>']

        # Encode each sequence using the vocabulary and pad to target length
        padded_seqs = [
            F.pad(torch.tensor([vocab_dict.get(c, pad_idx) for c in chars], dtype=torch.long),
                  (0, self.target_length - len(chars)),
                  value=pad_idx)
            for chars in sequences
        ]

        return padded_seqs, vocab_dict

    def _encode_labels(self):
        """
        Encode cell-level labels from categorical strings to integer codes.
        """
        categories = pd.Categorical(self.adata.obs['label'])  # Convert to categorical
        label_mapping = {cat: idx for idx, cat in enumerate(categories.categories)}  # Label → ID mapping
        labels = torch.tensor(categories.codes, dtype=torch.long)  # Integer-encoded label tensor
        return labels, label_mapping

    def get_smile_seqs(self):
        """Return encoded TCR sequences and the vocabulary."""
        return self.smile_seqs, self.vocab_dict

    def get_label(self):
        """Return labels and label-to-index mapping."""
        return np.array(self.label), self.label_mapping

    def __getitem__(self, index):
        # TCR_cdr3, gene_genematrix, TCR_ids, y, index
        return (
            self.smile_seqs[index],
            self.genematrix[index],
            self.TCR_ids[self.tcrcdr3.iloc[index]],
            self.label[index],
            index
                )

    def __len__(self):
        return len(self.smile_seqs)

class PredictDataset(Dataset):
    def __init__(self, adata, target_length=25):
        """
        Dataset for pretraining using TCR sequences, gene expression, and metadata (GEO, type).
        """
        self.adata = adata
        if sp.issparse(adata.X):
            self.genematrix = np.array(adata.X.toarray(), dtype=np.float32)
        else:
            self.genematrix = np.array(adata.X, dtype=np.float32)
        self.tcrcdr3 = adata.obs['cdr3']
        self.target_length = target_length

        self.smile_seqs, self.vocab_dict = self._encode_tcr()
        self.TCR_ids = {seq: idx for idx, seq in enumerate(self.tcrcdr3.unique())}

        self.GEOlabel = adata.obs['orig.ident']
        # self.type = adata.obs['celltypist_labels']
        # self.cdr3 = adata.obs['cdr3']

    def _encode_tcr(self):
        """
        Tokenize and pad CDR3 sequences into fixed-length tensors.
        """
        vocab = {}
        sequences = []

        for seq in self.tcrcdr3:
            chars = list(seq)
            sequences.append(chars)
            for c in chars:
                vocab[c] = vocab.get(c, 0) + 1

        vocab_dict = {c: i + 1 for i, (c, _) in enumerate(sorted(vocab.items(), key=lambda x: x[1], reverse=True))}
        vocab_dict['<pad>'] = 0
        pad_idx = vocab_dict['<pad>']

        padded_seqs = [
            F.pad(torch.tensor([vocab_dict.get(c, pad_idx) for c in chars], dtype=torch.long),
                  (0, self.target_length - len(chars)),
                  value=pad_idx)
            for chars in sequences
        ]

        return padded_seqs, vocab_dict

    def __getitem__(self, index):
        """
        Return a single training sample including:
        - Padded TCR sequence
        - Gene expression vector
        - TCR ID
        - Cell index
        - GEO label
        - type label
        """
        return (
            # TCR_beta, gene_matrix, TCR_ids, index, GEO_label, type_label, beta_label
            self.smile_seqs[index],
            self.genematrix[index],
            self.TCR_ids[self.tcrcdr3.iloc[index]],
            index,
            # self.GEOlabel.iloc[index]
            # self.type.iloc[index],
            # self.cdr3.iloc[index]
        )

    def __len__(self):
        return len(self.smile_seqs)

    def get_smile_seqs(self):
        return self.smile_seqs, self.vocab_dict
