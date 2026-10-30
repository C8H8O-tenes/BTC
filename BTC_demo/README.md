# BTC demo: graph construction and prediction

Run the demo in the configured BTC environment using matched single-cell RNA counts and TCR annotations. Graph construction and prediction run separately, so existing graphs can be reused for prediction.

## Environment

Run the following commands from the `BTC_demo` directory. Reference dependencies are provided in [BTC_environment.yml](BTC_environment.yml) and [requirements.txt](requirements.txt). Choose one installation option.

### Option 1: Conda environment

```bash
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
conda env create -f BTC_environment.yml
conda activate BTC_GNN
```

### Option 2: Install pip dependencies in a new environment

```bash
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
conda create -n BTC_GNN python=3.10 pip
conda activate BTC_GNN
python -m pip install -r requirements.txt
```

The additional package index above supplies the pinned `+cu118` packages. The shell commands and Conda package builds target Linux/WSL2. Compatible Python, PyTorch, and CUDA versions may also be used with the corresponding dependencies and installation source.

### Open the notebooks

From the `BTC_demo` directory:

```bash
python -m ipykernel install --user --name BTC_GNN --display-name BTC_GNN
cd notebooks
jupyter lab quickstart.ipynb
```

Run both notebooks from `BTC_demo/notebooks` and select the `BTC_GNN` kernel. Open `quickstart.ipynb` for graph construction and prediction, or `prepare_10x.ipynb` for input preparation.

## 1. Construct graphs with creat_graph.py

Run the following commands from the `BTC_demo` directory after activating the environment:

```bash
python creat_graph.py examples/sample_graph_input.h5ad --output-dir graphs --target-size 50 --device cuda:0
```

The script runs scGPT embedding, SEACells, Hallmark scoring, SCEPTR encoding, and heterogeneous graph construction. Outputs include `graphs/graph_manifest.csv`, graph files, MetaCell and MetaTCR metadata, and cell mappings.

## 2. Predict with byster.py

From the `BTC_demo` directory:

```bash
python byster.py --graph-dir graphs --output-dir predictions --device cuda:0
```

The default checkpoint is `models/btc/best_finetuned_model.pt`. Select another compatible checkpoint with `--checkpoint /path/to/best_finetuned_model.pt`.

The model predicts MetaTCR states and maps the results back to individual cells. `BTC_model.py` contains the `HeteroRelationLayer` and `HeteroFinetuner` classes extracted from the original `database_label_finetune/finetune_core.py`, preserving the model architecture and checkpoint format.

Both scripts accept `--device cpu`. Omitting `--device` automatically selects an available CUDA device or CPU. Choose new or empty output directories. To repeat prediction with existing graphs, choose another `--output-dir`.

Bundled resources are located relative to the scripts. User-supplied input and output paths are resolved relative to the current working directory.

## Use your own data

From the `BTC_demo` directory:

```bash
python creat_graph.py /path/to/my_data.h5ad --output-dir my_graphs --device cuda:0
python byster.py --graph-dir my_graphs --output-dir my_predictions --device cuda:0
```

Provide an AnnData (`.h5ad`) file with matched RNA and TCR annotations for each cell.

| Location | Content |
|---|---|
| `layers['counts']` or `X` | Nonnegative raw integer RNA counts; the counts layer takes priority by default. |
| `var_names` | Unique gene symbols, such as CD3D and TRAC. |
| `obs_names` | Unique cell IDs. |
| `obs['sample_id']` | Sample ID; each sample belongs to one patient. |
| `obs['patient_id']` | Patient ID. |
| `obs['cdr3_aa_a']`, `obs['cdr3_aa_b']` | TRA/TRB CDR3 amino-acid sequences, with at least one chain available per retained cell. |
| `obs['v_gene_a']`, `obs['j_gene_a']`, `obs['v_gene_b']`, `obs['d_gene_b']`, `obs['j_gene_b']` | Optional V/D/J annotations. |

Apply cell-quality filtering before graph construction. `--tcr-mode any` retains cells with at least one available chain; `--tcr-mode paired` retains cells with both chains. Predictions cover the retained cells. Select the intended cell population before running the workflow.

Graph-construction options include `--sample-col`, `--patient-col`, `--counts-layer`, `--batch-size`, `--threads`, `--target-size`, `--knn`, and `--seed`. Defaults include `--target-size 50`, `--knn 10`, and `--seed 42`.

For all options:

```bash
python creat_graph.py --help
python byster.py --help
```

Hallmark scores are standardized using the mean and standard deviation of the current input dataset. Keep the input cell population consistent when comparing predictions across runs.

## Prediction outputs

| File | Content |
|---|---|
| `metatcr_predictions.csv` | MetaTCR IDs, TCR annotations, logits, probabilities, and predicted labels. |
| `cell_predictions.csv` | Cell IDs, sample and patient IDs, MetaTCR assignments, and mapped predictions. |
| `qc_summary.json` | Graph, MetaTCR, and predicted-cell counts; threshold, checkpoint information, and execution status. |
| `run_config.json` | Prediction parameters and paths. |
| `error.log` | Diagnostic details for a failed prediction run. |

`probability` is the model-predicted probability of the tumor-specific class. By default, probabilities >= 0.5 are labeled `tumor-specific`; lower values are labeled `bystander`. Adjust the cutoff with `--threshold`. Cells assigned to the same MetaTCR inherit the same probability and label.

## Files and notebooks

| Path | Content |
|---|---|
| `BTC_environment.yml`, `requirements.txt` | Reference environment dependencies. |
| `creat_graph.py` | Standalone graph-construction script. |
| `byster.py` | Standalone prediction script. |
| `BTC_model.py` | BTC model classes. |
| `models/` | BTC checkpoint and scGPT model assets. |
| `resources/` | Hallmark gene sets, segment vocabulary, and asset checksums. |
| `examples/` | Example input data. |
| [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) | Graph construction and prediction in separate steps. |
| [notebooks/prepare_10x.ipynb](notebooks/prepare_10x.ipynb) | Preparation of matched RNA/TCR inputs from 10x files. |

The bundled BTC checkpoint originates from `sample/data/DATABASE/best_finetuned_model.pt` and uses the last-layer strategy at epoch 6. The example dataset is MSK1263_R1 and contains 6,982 cells. This patient contributed to checkpoint training; use the example to demonstrate the workflow. Evaluate generalization on an independent dataset. SCEPTR weights are supplied by the installed SCEPTR package.

## Package the demo

From the `BTC_demo` directory:

```bash
python scripts/package.py
```

This creates `dist/BTC_demo_full.zip` with code, models, example data, and resources. The archive excludes generated graphs, predictions, and run outputs.

Asset checksums are recorded in [resources/assets.json](resources/assets.json). Source information is recorded in [PROVENANCE.md](PROVENANCE.md).
