"""Integration checks using the bundled checkpoint and a real generated graph."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from byster import predict, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_dir', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = ROOT/'models/btc/best_finetuned_model.pt'
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        summary = predict(args.graph_dir, checkpoint, output, 'cpu', 0.5)
        meta = pd.read_csv(output/'metatcr_predictions.csv')
        cells = pd.read_csv(output/'cell_predictions.csv')
        assert cells.cell_id.is_unique
        assert cells.groupby('metatcr_id').probability.nunique().eq(1).all()
        assert cells.probability.between(0,1).all()
        assert meta.metatcr_id.is_unique
        assert len(cells) == summary['predicted_cells']
        assert np.where(cells.probability >= .5,'tumor-specific','bystander').tolist() == cells.predicted_label.tolist()
        manifest = pd.read_csv(args.graph_dir/'graph_manifest.csv')
        graph = torch.load(args.graph_dir/manifest.filename.iloc[0],weights_only=False,map_location='cpu')
        model, _ = load_model(checkpoint, 'cpu')
        with torch.inference_mode():
            single = model(graph['node_features'],graph['edge_index'],graph['edge_weight']).sigmoid().numpy()
        assert np.allclose(single,meta.sort_values('metatcr_id').probability.to_numpy(),atol=1e-6)
        bad = output/'bad_graphs'
        bad.mkdir()
        manifest.to_csv(bad/'graph_manifest.csv',index=False)
        graph['node_features']['metacell'] = graph['node_features']['metacell'][:,:-1]
        torch.save(graph,bad/manifest.filename.iloc[0])
        try:
            predict(bad,checkpoint,output,'cpu',0.5)
        except ValueError as error:
            assert 'invalid metacell features' in str(error)
        else:
            raise AssertionError('Incompatible graph was accepted')
        print(json.dumps(summary,indent=2))
        print('PASS: checkpoint load, probabilities, coverage, cell mapping, threshold and direct inference parity')

if __name__ == '__main__': main()
