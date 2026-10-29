"""BTC inference for MetaTCRs and mapping of predictions to cells."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from BTC_model import HeteroFinetuner


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    dims, relations = checkpoint['input_dims'], checkpoint['relations']
    state = checkpoint['model_state']
    hidden = state['input_projection.metacell.0.weight'].shape[0]
    layers = 1 + max(int(k.split('.')[1]) for k in state if k.startswith('layers.'))
    model = HeteroFinetuner(dims, relations, hidden, layers)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def predict(graph_dir, checkpoint_path, output, device, threshold):
    model, checkpoint = load_model(checkpoint_path, device)
    dims, relations = checkpoint['input_dims'], checkpoint['relations']
    manifest = pd.read_csv(graph_dir / 'graph_manifest.csv', dtype={'graph_id': str, 'patient_id': str})
    if manifest.empty or not manifest.graph_id.is_unique:
        raise ValueError('Graph manifest must contain unique graphs')
    rows = []
    with torch.inference_mode():
        for record in manifest.itertuples(index=False):
            graph = torch.load(graph_dir / record.filename, map_location='cpu', weights_only=False)
            if list(graph['node_features']) != list(dims) or list(graph['edge_index']) != list(relations):
                raise ValueError(f'{record.graph_id}: checkpoint/graph schema mismatch')
            for node, dim in dims.items():
                x = graph['node_features'][node]
                if tuple(x.shape) != (graph['num_nodes'][node], dim) or not torch.isfinite(x).all():
                    raise ValueError(f'{record.graph_id}: invalid {node} features')
            logits = model(*[{k: v.to(device) for k,v in graph[field].items()} for field in ['node_features','edge_index','edge_weight']])
            scores = logits.sigmoid().cpu().numpy()
            ids, labels = graph['mappings']['metatcr_global_ids'], graph['mappings']['metatcr_labels']
            if len(scores) != len(ids) or len(labels) != len(ids) or not np.isfinite(scores).all():
                raise ValueError('Invalid prediction length or probability')
            rows.extend(dict(patient_id=record.patient_id, sample_id=record.graph_id, metatcr_id=int(i), metatcr_label=str(label), logit_tumor_specific=float(logit), probability=float(score)) for i,label,logit,score in zip(ids,labels,logits.cpu().numpy(),scores))
    predictions = pd.DataFrame(rows).sort_values('metatcr_id')
    if not predictions.metatcr_id.is_unique or len(predictions) != manifest.n_metatcr.sum():
        raise ValueError('MetaTCR coverage mismatch')
    metadata = pd.read_csv(graph_dir / 'metatcr_metadata.csv', keep_default_na=False)
    if set(metadata.metatcr_id) != set(predictions.metatcr_id):
        raise ValueError('MetaTCR metadata coverage mismatch')
    predictions = predictions.merge(metadata.drop(columns=[c for c in metadata if c in predictions and c != 'metatcr_id']), on='metatcr_id', validate='one_to_one')
    predictions['predicted_label'] = np.where(predictions.probability >= threshold, 'tumor-specific', 'bystander')
    mapping = pd.read_csv(graph_dir / 'cell_to_metatcr.csv', dtype={'cell_id':str,'sample_id':str})
    if not mapping.cell_id.is_unique:
        raise ValueError('Duplicate cell IDs in mapping')
    cells = mapping.merge(predictions[['metatcr_id','sample_id','patient_id','probability','predicted_label']], on=['metatcr_id','sample_id'], how='left', validate='many_to_one')
    if cells.probability.isna().any():
        raise ValueError('Unmapped retained cells')
    predictions.to_csv(output / 'metatcr_predictions.csv', index=False)
    cells.to_csv(output / 'cell_predictions.csv', index=False)
    return dict(graphs=len(manifest), metatcrs=len(predictions), predicted_cells=len(cells), tumor_specific_cells=int(cells.predicted_label.eq('tumor-specific').sum()), positive_class='tumor-specific', threshold=threshold, prediction_unit='MetaTCR; cell probabilities inherited through cell_to_metatcr.csv', checkpoint_epoch=int(checkpoint['epoch']), checkpoint_strategy=checkpoint['strategy'])


def main():
    import argparse
    import json
    import traceback
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="BTC model inference on graphs produced by creat_graph.py")
    parser.add_argument('--graph-dir', type=Path, default=Path('graphs'))
    parser.add_argument('--checkpoint', type=Path, default=root/'models/btc/best_finetuned_model.pt')
    parser.add_argument('--output-dir', type=Path, default=Path('predictions'))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be in [0, 1]')
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    for path in [args.checkpoint, args.graph_dir/'graph_manifest.csv',
                 args.graph_dir/'metatcr_metadata.csv', args.graph_dir/'cell_to_metatcr.csv']:
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'run_config.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
    summary = dict(status='running')
    summary_path = args.output_dir/'qc_summary.json'
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    try:
        summary.update(predict(args.graph_dir,args.checkpoint,args.output_dir,args.device,args.threshold))
        summary['status'] = 'complete'
        print(json.dumps(summary,indent=2),flush=True)
        print(f'BTC predictions saved to {args.output_dir.resolve()}',flush=True)
    except BaseException as error:
        summary.update(status='failed',error=str(error))
        (args.output_dir/'error.log').write_text(traceback.format_exc())
        raise
    finally:
        summary_path.write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
