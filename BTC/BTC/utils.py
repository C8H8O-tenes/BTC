import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import math
import logging
import yaml
import random
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nt_xent import NT_Xent

# --------------------------- Functions --------------------------- #
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    for handler in [logging.FileHandler(filename, "w"), logging.StreamHandler()]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pretrained_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

# position embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
# attention
def get_attn_pad_mask(seq_q, seq_k, pad_id=0):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    # return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    key_mask   = seq_k.eq(pad_id).unsqueeze(1).expand(batch_size, len_q, len_k)   # 屏蔽列
    query_mask = seq_q.eq(pad_id).unsqueeze(2).expand(batch_size, len_q, len_k)   # 屏蔽行
    return (key_mask | query_mask).bool()
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)  # [batch_size, seq_len, d_model]
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        # 这里计算注意力
        attn = nn.Softmax(dim=-1)(scores)
        attn_score = self.calculate(attn)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn, attn_score

    def calculate(self,attn):
        score = torch.sum(attn, dim=2)
        return score
        # print(score,score.shape)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn, attn_score = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        # print(attn)
        return self.layer_norm(output + residual), attn, attn_score

def creat_mask_matrix(mask, smiles, precet):
    """
    mask (batch_size, src_len): 权重矩阵
    smiles (batch_size, src_len): 输入张量
    precet: 掩码比例
    """
    batch_size, src_len = smiles.shape
    one = torch.ones_like(smiles, dtype=torch.int32, device=smiles.device)

    # 根据权重矩阵筛选掩码位置
    k = int(math.ceil(precet * src_len))
    _, indices = torch.topk(mask, k, dim=1, largest=False, sorted=False)  # 选出最小的 k 个位置
    one.scatter_(1, indices, 0)  # 将这些位置掩码置为 0

    # 应用掩码矩阵
    re = smiles * one
    return re
def create_masked_seq(scores, seq, precet, pad_id=0, mask_id=None):
    assert 0.0 <= precet <= 1.0
    B, L = seq.shape
    out = seq.clone()
    valid = (seq != pad_id)
    repl = pad_id if mask_id is None else mask_id
    for b in range(B):
        vb = valid[b]
        n  = int(vb.sum().item())
        if n == 0:
            continue
        k  = max(1, math.ceil(n * precet))
        # PAD as inf
        s = scores[b].clone().masked_fill(~vb, float('inf'))
        idx = torch.topk(s, k, largest=False, sorted=False).indices
        out[b, idx] = repl
    return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.enc_self_attn = MultiHeadAttention(self.d_model, self.d_k, self.d_v, self.n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # encoded_TCR: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        encoded_TCR, attn, attn_score = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        encoded_TCR = self.pos_ffn(encoded_TCR)  # encoded_TCR: [batch_size, src_len, d_model]
        return encoded_TCR, attn, attn_score
# encoder
class TCR_encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_ff, self.d_k, self.d_v, self.n_heads) for _ in range(self.n_layers)])

    def forward(self, enc_inputs, encoded_TCR):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        enc_self_attns_score = []
        for layer in self.layers:
            # encoded_TCR: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            encoded_TCR, enc_self_attn, attn_score = layer(encoded_TCR, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
            # print(attn_score) 10*2*26a
            enc_self_attns_score.append(torch.sum(attn_score, dim=1))
        return encoded_TCR, enc_self_attns, enc_self_attns_score

class ProfileAutoEncoder(nn.Module):
    def __init__(self, rna_in_dim, rna_hid_dim, rna_hid_dim2, out_dim, dropout, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(rna_in_dim, rna_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rna_hid_dim, rna_hid_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rna_hid_dim2, out_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, rna_hid_dim2),
            nn.ReLU(),
            nn.Linear(rna_hid_dim2, rna_hid_dim),
            nn.ReLU(),
            nn.Linear(rna_hid_dim, rna_in_dim)
        )

    def encode(self, x, add_noise: bool = True):
        if self.training and add_noise and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        z = self.encoder(x)
        return z

    def forward(self, x, add_noise: bool = True):
        z = self.encode(x, add_noise=add_noise)
        recon = self.decoder(z)
        return z, recon

class ProfileVAE(nn.Module):
    def __init__(self, rna_in_dim, rna_hid_dim, rna_hid_dim2, out_dim, dropout):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(rna_in_dim, rna_hid_dim),
            nn.LayerNorm(rna_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rna_hid_dim, rna_hid_dim2),
            nn.LayerNorm(rna_hid_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc_mu = nn.Linear(rna_hid_dim2, out_dim)
        self.fc_logvar = nn.Linear(rna_hid_dim2, out_dim)

        self.decoder = nn.Sequential(
            nn.Linear(out_dim, rna_hid_dim2),
            nn.ReLU(),
            nn.Linear(rna_hid_dim2, rna_hid_dim),
            nn.ReLU(),
            nn.Linear(rna_hid_dim, rna_in_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return z, recon, mu, logvar

class BTC(nn.Module):
    def __init__(self,
                 d_model, d_ff, d_k, d_v, n_heads, n_layers,
                 precet, seq_len,
                 rna_in_dim, rna_hid_dim, rna_hid_dim2, out_dim,
                 dropout, temperature, number_of_type = 5
                 ):
        super().__init__()
        self.post_ln = nn.LayerNorm(out_dim)

        self.seq_len = seq_len
        self.dropout = dropout
        self.precet = precet
        self.temperature = temperature

        # ====================== RNA Encoder ======================
        self.encoder_profile = ProfileAutoEncoder(rna_in_dim, rna_hid_dim, rna_hid_dim2, out_dim, dropout)
        # self.encoder_profile = ProfileVAE(rna_in_dim, rna_hid_dim, rna_hid_dim2, out_dim, dropout)

        self.rna_contrastive_head = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, out_dim, bias=False)
        )

        # ====================== TCR Encoder ======================
        self.pad_id = 0
        self.src_emb = nn.Embedding(self.seq_len, d_model, padding_idx=self.pad_id)
        self.pos_emb = PositionalEncoding(d_model)
        self.pre_mask_attention = TCR_encoder(d_model, d_ff, d_k, d_v, n_heads, n_layers)
        self.encoder_TCR = TCR_encoder(d_model, d_ff, d_k, d_v, n_heads, n_layers)

        self.tcr_embedding_head = nn.Sequential(
            nn.Linear(d_model, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim // 2, out_dim)
        )

        self.tcr_contrastive_head = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, out_dim, bias=False)
        )

        self.NT_Xent = NT_Xent

        # ====================== Fusion Module ======================
        self.gate_tau = nn.Parameter(torch.tensor(10.0))
        self.fusion_gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, 1)
        )

        self.dec_rna = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.dec_tcr = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def build_finetune_head(self, out_dim):
        self.ptm_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_dim // 2, 1)
        )
        # RNA-only 分类头
        self.rna_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_dim // 2, 1)
        )

    def forward(self, tcr, rna):
        def encode_full(seq, valid):
            x = self.src_emb(seq)
            x = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
            x = x * valid.unsqueeze(-1).float()
            encoded, attn, attn_score = self.encoder_TCR(seq, x)  # attn: list[n_layers] of [B,nH,L,L]
            return encoded, attn, attn_score

        # calculate mask strategy
        valid = (tcr != self.pad_id)

        emb = self.src_emb(tcr)
        emb = self.pos_emb(emb.transpose(0, 1)).transpose(0, 1)
        emb = emb * valid.unsqueeze(-1).float()
        _, mask_attn, _ = self.pre_mask_attention(tcr, emb)  # list[len=n_layers], each [B, nH, L, L]
        att = torch.stack(mask_attn, dim=0).float()

        qmask = valid.unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # [1,B,1,L,1]
        kmask = valid.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1,B,1,1,L]
        att = att * qmask * kmask

        scores = att.mean(0).mean(1).sum(1)
        mask_tcr = create_masked_seq(scores, tcr, self.precet, pad_id=self.pad_id, mask_id=None)

        # smiles and mask_smiles
        enc_tcr, _, attn_score = encode_full(tcr, valid)
        enc_tcr_m, _, _ = encode_full(mask_tcr, valid)

        valid_a = (tcr != self.pad_id).float().unsqueeze(-1)  # [B,L,1]
        valid_b = (mask_tcr != self.pad_id).float().unsqueeze(-1)  # [B,L,1]
        den_a = valid_a.sum(1).clamp_min(1.0)
        den_b = valid_b.sum(1).clamp_min(1.0)
        enc_tcr = (enc_tcr * valid_a).sum(1) / den_a  # [B, d_model]
        enc_tcr_m = (enc_tcr_m * valid_b).sum(1) / den_b

        # extract tcr feature
        tcr_embed = self.tcr_embedding_head(enc_tcr)
        tcr_embed_mask = self.tcr_embedding_head(enc_tcr_m)

        # contrastive learning
        z1_tcr = F.normalize(self.tcr_contrastive_head(tcr_embed), dim=1)
        z2_tcr = F.normalize(self.tcr_contrastive_head(tcr_embed_mask), dim=1)

        contrastive_loss_fn = self.NT_Xent(tcr_embed.shape[0], self.temperature, world_size=1)
        tcr_contrast_loss = contrastive_loss_fn(z1_tcr, z2_tcr)

        ############### RNA reconstruction
        rna_embed, rna_recon = self.encoder_profile(rna)
        rna_recon_loss = F.mse_loss(rna_recon, rna) + 0.1 * torch.mean(torch.abs(rna_recon - rna))

        mask_rna = mask_rna_input(rna, mask_ratio=0.15)
        mask_rna_embed, *_ = self.encoder_profile(mask_rna)

        z1_rna = F.normalize(self.rna_contrastive_head(rna_embed), dim=1)
        z2_rna = F.normalize(self.rna_contrastive_head(mask_rna_embed), dim=1)
        rna_contrast_fn = self.NT_Xent(rna_embed.size(0), self.temperature,
                                       world_size=1)
        rna_contrast_loss = rna_contrast_fn(z1_rna, z2_rna)
        rna_loss = rna_recon_loss + 0.3 * rna_contrast_loss

        # ----- Fusion -----
        tcr_e = self.post_ln(tcr_embed)
        rna_e = self.post_ln(rna_embed)

        gate_input = torch.cat([rna_e, tcr_e], dim=1)  # 用对齐尺度后的向量
        gate_input = gate_input + 1e-3 * torch.randn_like(gate_input)
        logit = self.fusion_gate(gate_input)  # [B, 1]
        alpha = 0.8 + 0.1 * torch.tanh(logit / self.gate_tau)
        # print("alpha mean/std:", alpha.mean().item(), alpha.std().item())
        z_joint = alpha * rna_e + (1 - alpha) * tcr_e

        recon_rna = self.dec_rna(z_joint)
        recon_tcr = self.dec_tcr(z_joint)
        cos_r = 1 - F.cosine_similarity(recon_rna, rna_e.detach(), dim=1).mean()
        cos_t = 1 - F.cosine_similarity(recon_tcr, tcr_e.detach(), dim=1).mean()
        mag_r = F.mse_loss(recon_rna.norm(dim=1), rna_e.detach().norm(dim=1))
        mag_t = F.mse_loss(recon_tcr.norm(dim=1), tcr_e.detach().norm(dim=1))
        fusion_loss = (cos_r + cos_t) + 0.1 * (mag_r + mag_t)

        total_loss = (
                0.5 * tcr_contrast_loss +
                1.0 * rna_loss +
                1.0 * fusion_loss
        )

        return total_loss, tcr_contrast_loss, rna_loss, fusion_loss, z_joint, attn_score, tcr_embed, rna_embed

    def pretrain_forward(self, tcr, rna):
        total_loss, contrast_loss, recon_loss, fusion_loss, z_joint, attn_score, tcr_embed, rna_embed = self.forward(tcr, rna)
        return total_loss, contrast_loss, recon_loss, fusion_loss, z_joint, attn_score, tcr_embed, rna_embed

    def finetune_forward(self, tcr, rna, label):
        total_loss, tcr_loss, rna_loss, fusion_loss, z_joint, attn_score, tcr_embed, rna_embed = self.forward(tcr, rna)

        pred = self.ptm_head(z_joint) # logits (no sigmoid)
        pred_rna = self.rna_head(rna_embed)  # [B,1] logits

        criterion = nn.BCEWithLogitsLoss()
        class_loss = criterion(pred.squeeze(-1), label.float())
        class_loss_r = criterion(pred_rna.squeeze(-1), label.float())

        return total_loss, tcr_loss, rna_loss, fusion_loss, class_loss, class_loss_r, z_joint, attn_score, rna_embed, tcr_embed, pred

    def predict_forward(self, tcr, rna):
        contrast_loss, recon_loss, fusion_loss, z_joint, attn_score, tcr_embed, rna_embed = self.forward(tcr, rna)

        # class
        pred = self.z_joint_project(z_joint)
        for h in self.ptm_head:
            pred = h(pred)

        prob = torch.sigmoid(pred.squeeze(-1))

        return prob, z_joint, attn_score, rna_embed, tcr_embed

def mask_rna_input(rna, mask_ratio=0.15):
    B, D = rna.shape
    num_mask = int(D * mask_ratio)

    rna_masked = rna.clone()
    for i in range(B):
        idx = torch.randperm(D)[:num_mask]
        rna_masked[i, idx] = 0.0  # 遮盖部分设为0（也可以设为其他值）
    return rna_masked

def init_model(config, device):
    model = BTC(
        d_model=config['Train']['Model_Parameter']['d_model'],
        d_ff=config['Train']['Model_Parameter']['d_ff'],
        d_k=config['Train']['Model_Parameter']['d_k'],
        d_v=config['Train']['Model_Parameter']['d_v'],
        n_heads=config['Train']['Model_Parameter']['n_heads'],
        n_layers=config['Train']['Model_Parameter']['n_layers'],
        precet=config['Train']['Model_Parameter']['precet'],
        seq_len=25,

        rna_in_dim=config['Train']['Model_Parameter']['encoderprofile_in_dim'],
        rna_hid_dim=config['Train']['Model_Parameter']['encoderprofile_hid_dim'],
        rna_hid_dim2=config['Train']['Model_Parameter']['encoderprofile_hid_dim2'],
        out_dim=config['Train']['Model_Parameter']['encoderprofile_out_dim'],
        dropout=0.2,
        temperature=0.1
    ).to(device)

    model.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['Train']['Trainer_parameter']['learning_rate']
    )

    return model

def save_best_model(model, epoch, output_dir, stage=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }, f"{output_dir}/Model_checkpoints/merge_{stage}_epoch{epoch}_model.pth")

def train_one_epoch(model, dataloader, device):
    ##### embedding ############
    all_embeddings,all_rna_embeddings, all_tcr_embeddings = [], [], []
    all_GEO_labels, all_beta_labels, all_type_labels = [], [], []

    model.train()
    total_loss, loss_components = 0, {'tcr': 0, 'rna': 0, 'fusion': 0}

    for TCR_beta, gene_matrix, TCR_ids, batch_index, GEO_label, type_label, beta_label in tqdm(dataloader, desc="Pretrain", leave=False):
        loss, tcr_loss, rna_loss, fusion_loss, z_joint, attn_score, tcr_embed, rna_embed = model.pretrain_forward(
            TCR_beta.to(device), gene_matrix.to(device)
        )

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        total_loss += loss.item()
        loss_components['tcr'] += tcr_loss.item()
        loss_components['rna'] += rna_loss.item()
        loss_components['fusion'] += fusion_loss.item()

        ##### embedding ############
        all_embeddings.append(z_joint.detach().cpu().numpy())
        all_rna_embeddings.append(rna_embed.detach().cpu().numpy())
        all_tcr_embeddings.append(tcr_embed.detach().cpu().numpy())

        all_GEO_labels.extend(GEO_label)
        all_beta_labels.extend(beta_label)
        all_type_labels.extend(type_label)
        ##### embedding ############

    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}

    return avg_loss, avg_components, all_embeddings, all_rna_embeddings, all_tcr_embeddings, all_GEO_labels, all_beta_labels, all_type_labels

def finetune_one_epoch(model, dataloader, device, epoch):
    model.train()
    total_loss, loss_components = 0, {'tcr': 0, 'rna': 0, 'fusion': 0, 'class': 0}
    joint_embeds, rna_embeds, tcr_embeds = [], [], []
    labels, joint_logits, rna_logits = [], [], []

    if epoch < 10:
        for name, p in model.named_parameters():
            if name.startswith('encoder_profile') or name.startswith('rna_head') or name.startswith('ptm_head'):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = True

    for TCR_beta, gene_matrix, _, label, batch_index in tqdm(dataloader, desc="Finetune", leave=False):
        TCR_beta = TCR_beta.to(device)
        gene_matrix = gene_matrix.to(device)
        label = label.to(device).float()

        total_loss_batch, tcr_loss, rna_loss, fusion_loss, class_loss, class_loss_r, z_joint, attn_score, rna_embed, tcr_embed, pred = model.finetune_forward(
            TCR_beta, gene_matrix, label)

        joint_embeds.append(z_joint.detach().cpu().numpy())
        rna_embeds.append(rna_embed.detach().cpu().numpy())
        tcr_embeds.append(tcr_embed.detach().cpu().numpy())

        joint_logits.append(pred.squeeze(-1).detach().cpu().numpy())
        # rna_logits.append(rna_logit.detach().cpu().numpy())

        labels.append(label.cpu().numpy())

        loss = 5.0 * class_loss + 0.1 * tcr_loss + 0.1 * rna_loss + 0.1 * fusion_loss # 三瓣
        # loss = class_loss + class_loss_r + 0.1 * fusion_loss # all 分的很好 rna一团
        # loss = class_loss + class_loss_r + 0.5 * rna_loss + 0.1 * fusion_loss
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        total_loss += loss.item()
        loss_components['tcr'] += tcr_loss.item()
        loss_components['rna'] += rna_loss.item()
        loss_components['fusion'] += fusion_loss.item()
        loss_components['class'] += class_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    return avg_loss, avg_components

# def finetune_one_epoch(model, dataloader, device, lr_enc=1e-4, lr_head=1e-3):
#     # 1. 冻结 / 解冻
#     for p in model.parameters():
#         p.requires_grad = False
#     for p in model.encoder_profile.parameters():
#         p.requires_grad = True
#     for p in model.rna_head.parameters():
#         p.requires_grad = True
#     optimizer = torch.optim.AdamW([
#         {'params': model.encoder_profile.parameters(), 'lr': lr_enc},
#         {'params': model.rna_head.parameters(),       'lr': lr_head},
#     ], weight_decay=1e-4)
#     model.train()
#
#     total_loss, loss_components = 0, {'tcr': 0, 'rna': 0, 'fusion': 0, 'class': 0}
#     all_rnaembeds = []
#     all_labels = []
#     all_z_joint = []
#     all_tcrembeds = []
#     for TCR_beta, gene_matrix, _, label, _ in tqdm(dataloader):
#         TCR_beta = TCR_beta.to(device)
#         gene_matrix = gene_matrix.to(device)
#         label = label.to(device).float()
#
#         # —— 2. 一次前向拿所有 loss ——
#         # total_loss_batch, tcr_loss, rna_loss, fusion_loss, class_loss, class_loss_r, z_joint, attn_score, rna_embed, tcr_embed, pred = (
#         #     model.finetune_forward(TCR_beta, gene_matrix, label)
#         # )
#
#         rna_embed, _ = model.encoder_profile(gene_matrix)    # [B, D]
#         logits = model.rna_head(rna_embed).squeeze(-1)        # [B]
#         loss = F.binary_cross_entropy_with_logits(logits, label)
#
#         # model.optimizer.zero_grad()
#         # loss.backward()
#         # model.optimizer.step()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         # loss_components['tcr'] += tcr_loss.item()
#         # loss_components['rna'] += rna_loss.item()
#         # loss_components['fusion'] += fusion_loss.item()
#         # loss_components['class'] += class_loss.item()
#
#         all_rnaembeds.append(rna_embed.detach().cpu().numpy())
#         # all_z_joint.append(z_joint.detach().cpu().numpy())
#         # all_tcrembeds.append(tcr_embed.detach().cpu().numpy())
#         all_labels.append(label.detach().cpu().numpy())
#
#     avg_loss = total_loss / len(dataloader)
#     avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
#     return avg_loss, avg_components

def validate(model, dataloader, device):
    model.eval()
    true_labels, pred_probs = [], []
    TCR_embeddings, Profile_embeddings, joint_embeddings, attn_scores = [], [], [], []

    with torch.no_grad():
        for TCR_beta, gene_matrix, _, label, _ in tqdm(dataloader, desc="Validation", leave=False):
            _, _, _, _, _, _, z_joint, attn_score, rna_embed, tcr_embed, preds = model.finetune_forward(
                TCR_beta.to(device), gene_matrix.to(device), label.to(device)
            )

            true_labels.extend(label.cpu().numpy())
            # sigmoid trans logits
            pred_probs.extend(torch.sigmoid(preds).cpu().numpy())

            TCR_embeddings.extend(tcr_embed)
            Profile_embeddings.extend(rna_embed)
            joint_embeddings.extend(z_joint)
            attn_avg = torch.mean(torch.stack(attn_score), dim=0)
            attn_scores.extend(attn_avg)

    return np.array(true_labels), np.array(pred_probs), TCR_embeddings, Profile_embeddings, joint_embeddings, attn_scores

def predict(model, dataloader, device):
    model.eval()
    all_probs, all_z_joints, all_attn_scores, all_rna_embeds, all_tcr_embeds  = [], [], [], [], []

    with torch.no_grad():
        for TCR_beta, gene_matrix, TCR_ids, batch_index in tqdm(dataloader, desc="predict", leave=False):
            dummy_label = torch.zeros(TCR_beta.size(0), device=device)
            _, _, _, _, _, _, z_joint, attn_score, rna_embed, tcr_embed, logits = \
                model.finetune_forward(
                    TCR_beta.to(device),
                    gene_matrix.to(device),
                    dummy_label.to(device)
                )

            # sigmoid
            probs = torch.sigmoid(logits).cpu().numpy()
            # attention
            attn_avg = torch.mean(torch.stack(attn_score), dim=0).cpu().numpy()

            all_probs.extend(probs)
            all_z_joints.extend(z_joint.cpu().numpy())
            all_attn_scores.extend(attn_avg)
            all_rna_embeds.extend(rna_embed.cpu().numpy())
            all_tcr_embeds.extend(tcr_embed.cpu().numpy())

    return all_probs, all_z_joints, all_attn_scores, all_rna_embeds, all_tcr_embeds

