import argparse
import scanpy as sc
import numpy as np
import json
from collections import Counter
from torch.utils.data import DataLoader
from scipy.sparse import issparse

from main_tcr_train import *
from merge_Dataset import *
from utils import *

def main(args):
    # 1. Load config
    config = load_config(args.config)
    set_random_seed(config['Train']['Trainer_parameter']['random_seed'])
    device = config['Train']['Model_Parameter']['device']

    # 2. Load data
    adata = sc.read_h5ad(args.input)
    if issparse(adata.X):
        adata.X = np.array(adata.X.toarray(), dtype=np.float32)
    else:
        adata.X = np.array(adata.X, dtype=np.float32)

    # --- Add this check ---
    is_counts = np.allclose(adata.X, np.round(adata.X)) and np.max(adata.X) > 50
    if is_counts:
        print("Detected raw counts matrix → running normalize + log1p + HVG")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.uns['log1p']['base'] = None  # 避免写 h5ad 出错
    else:
        print("Input looks normalized/logged → only selecting HVGs")

    # adata.uns['log1p']['base'] = None
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    adata = adata[:, adata.var['highly_variable']].copy()
    if adata.n_vars < 5000:
        raise ValueError(
            f"HVG selection failed: only {adata.n_vars} genes selected, "
            f"but 5000 required. Please provide a raw counts matrix as input."
        )
    else:
        print(f"✅ HVG selection complete: {adata.n_vars} genes retained.")

    # 3. Load model
    with open(args.label_map, "r") as f:
        label_mapping = json.load(f)
    index_to_label = {v: k for k, v in label_mapping.items()}

    model = init_model(config, device)
    model.build_finetune_head(128)
    model = load_pretrained_model(model, args.checkpoint, device)

    # 4. Predict
    predict_dataset = PredictDataset(adata)
    predict_loader = DataLoader(predict_dataset, batch_size=config['Train']['Sampling']['batch_size'], shuffle=False)
    prob, z_joint, attn_score, rna_embed, tcr_embed = predict(model, predict_loader, device)

    probs = np.array(prob)
    pred_labels = (probs > 0.5).astype(int)
    labels = [index_to_label[label.item()] for label in pred_labels]
    adata.obs['labels'] = labels

    # 6. Save output
    adata.write_h5ad(args.output)
    print(f"Prediction done! Saved to: {args.output}")
    print("Label counts:", Counter(labels))


def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Run BTC model prediction on an AnnData object."
    )
    p.add_argument('--config', type=str, required=True, help='Path to YAML config')
    p.add_argument('--input', type=str, required=True, help='Input .h5ad file')
    p.add_argument('--checkpoint', type=str, default="./TrainingResult/Model_checkpoints/lung_best_model.pth",
                   help='Model checkpoint (.pth), default: lung_best_model.pth')
    p.add_argument('--output', type=str, required=True, help='Output .h5ad file')
    p.add_argument('--label_map', type=str, default='label_mapping.json',
                   help='Path to label mapping JSON')
    return p

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(
        description="Run BTC model prediction on an AnnData object.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--input', type=str, required=True, help='Input .h5ad file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint (.pth)')
    parser.add_argument('--output', type=str, required=True, help='Output .h5ad file')
    parser.add_argument('--label_map', type=str, default='label_mapping.json',
                        help='Path to label mapping JSON')

    if "pydevd" in sys.modules and len(sys.argv) == 1:
        # 在 PyCharm 调试模式下，没有手动填参数时 → 给默认值
        args = parser.parse_args([
            "--config", "/data/hxu10/bystander/BTC/config.yaml",
            "--input", "../Data/blood_predicte/GSE235760_merged_celltypes.h5ad",
            "--checkpoint", "./TrainingResult/Model_checkpoints/lung_best_model.pth",
            "--output", "../Data/blood_predicte/GSE235760_merged_BTCpredict.h5ad",
            "--label_map", "label_mapping.json"
        ])
    else:
        args = parser.parse_args()

    main(args)




