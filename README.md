# BYSTER

**BYSTER: multimodal graph learning resolves bystander versus tumor-specific T cell states in the tumor immune microenvironment**

BYSTER is a multimodal graph learning framework for resolving bystander versus tumor-specific T cell states from matched single-cell RNA expression and TCR annotations. The workflow constructs heterogeneous graphs, predicts MetaTCR states using a pretrained model, and maps the predictions back to individual cells.

## Model overview

![BYSTER model architecture](readme/BYSTER_framework.png)

## Environment setup

Reference dependencies are provided in [BTC_environment.yml](readme/BTC_environment.yml) (Conda environment and pip dependencies) and [requirements.txt](readme/requirements.txt) (pip dependencies). Compatible versions, including newer releases that preserve the required APIs, may also be used. Choose Python, PyTorch, and CUDA versions appropriate for your system.

Run the installation commands from the repository root. The Conda YAML records Linux package builds; adapt the dependency files for your platform as needed. For the supplied `+cu118` packages, set the additional package index in a Linux/WSL2 shell:

```bash
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
```

### Option 1: Conda environment

Create the environment with its Conda and pip dependencies:

```bash
conda env create -f readme/BTC_environment.yml
conda activate BTC_GNN
```

### Option 2: Install pip dependencies in a new environment

```bash
conda create -n BTC_GNN python=3.10 pip
conda activate BTC_GNN
python -m pip install -r readme/requirements.txt
```

Choose one installation option. To open the tutorial, register the environment as a Jupyter kernel and start JupyterLab:

```bash
python -m ipykernel install --user --name BTC_GNN --display-name BTC_GNN
cd BTC_demo/notebooks
jupyter lab quickstart.ipynb
```

Run both notebooks from `BTC_demo/notebooks`. Open `quickstart.ipynb` for graph construction and prediction, or `prepare_10x.ipynb` for input preparation. Select the `BTC_GNN` kernel in JupyterLab.

## Tutorial

Start with [quickstart.ipynb](BTC_demo/notebooks/quickstart.ipynb) for graph construction and model inference. For matched 10x RNA and TCR files, see [prepare_10x.ipynb](BTC_demo/notebooks/prepare_10x.ipynb). The input preparation tutorial is also available in [new_dataset.ipynb](new_dataset.ipynb).

The tutorial includes:

- Input requirements for matched RNA expression and TCR annotations.
- An example H5AD dataset for running the workflow.
- Graph construction using scGPT, SEACells, Hallmark scores, and SCEPTR.
- Model inference and mapping of MetaTCR predictions to cells.
- Optional preparation of matched RNA and TCR inputs from 10x files.

## Quick start

From the repository root, enter `BTC_demo` and run both steps below in that directory after activating the environment.

### 1. Construct graphs

```bash
cd BTC_demo
python creat_graph.py examples/sample_graph_input.h5ad --output-dir graphs --target-size 50 --device cuda:0
```

### 2. Run prediction

```bash
python byster.py --graph-dir graphs --output-dir predictions --device cuda:0
```

Graph construction and prediction run separately. Reuse existing graphs for subsequent prediction runs and choose new or empty output directories. Both scripts accept `--device cpu`; omitting `--device` selects an available CUDA device or CPU automatically.

The default checkpoint is `models/btc/best_finetuned_model.pt` within `BTC_demo`. Use `--checkpoint` to select another checkpoint with the matching model architecture and graph schema.

### Prediction outputs

Results are written to `BTC_demo/predictions/`:

| File | Content |
|---|---|
| `metatcr_predictions.csv` | MetaTCR annotations, logits, probabilities, and predicted labels. |
| `cell_predictions.csv` | Cell IDs, sample and patient IDs, MetaTCR assignments, and mapped predictions. |
| `qc_summary.json` | Execution status, graph and prediction counts, threshold, and checkpoint information. |
| `run_config.json` | Prediction parameters and paths. |

`probability` represents the model-predicted probability of the tumor-specific class. By default, probabilities ≥ 0.5 are labeled `tumor-specific`; lower values are labeled `bystander`. Adjust the threshold with `--threshold`. Cells assigned to the same MetaTCR inherit the same probability and label.

## Input requirements

Provide an AnnData (`.h5ad`) file in which each row represents one cell. RNA expression and TCR annotations must refer to the same cell.

| Location | Requirement |
|---|---|
| `adata.layers['counts']` or `adata.X` | Nonnegative raw integer RNA counts, cells × genes. The loader uses `layers['counts']` when present, otherwise `X`. |
| `adata.var_names` | Unique gene symbols. |
| `adata.obs_names` | Unique cell IDs. Use sample prefixes when combining samples. |
| `adata.obs['patient_id']` | Nonmissing patient or donor ID. |
| `adata.obs['sample_id']` | Nonmissing sample or library ID. Each sample belongs to one patient. |
| `adata.obs['cdr3_aa_a']` | TRA CDR3 amino-acid sequence, when available. |
| `adata.obs['cdr3_aa_b']` | TRB CDR3 amino-acid sequence, when available. |

Provide at least one CDR3 column. `--tcr-mode any` retains cells with at least one available TRA or TRB CDR3 sequence; `--tcr-mode paired` retains cells with both chains.

Optional TCR gene annotations are `v_gene_a`, `j_gene_a`, `v_gene_b`, `d_gene_b`, and `j_gene_b`. Preserve valid annotations when available because they inform TCR encoding.

Apply cell-quality filtering before export and retain unnormalized RNA counts.

### Run on your own data

From the `BTC_demo` directory:

```bash
python creat_graph.py /path/to/my_data.h5ad --output-dir my_graphs --device cuda:0
python byster.py --graph-dir my_graphs --output-dir my_predictions --device cuda:0
```

For all available options, run `python creat_graph.py --help` and `python byster.py --help`.

## Optional preparation from 10x files

Follow [prepare_10x.ipynb](BTC_demo/notebooks/prepare_10x.ipynb) using matched files from the same cell library:

| File | Content |
|---|---|
| `barcodes.tsv.gz` | RNA cell barcodes |
| `features.tsv.gz` | RNA feature annotations |
| `matrix.mtx.gz` | RNA count matrix |
| `filtered_contig_annotations.csv.gz` | TCR contig annotations |

The preparation notebook writes the following files to `BTC_demo/outputs/prepared/`:

| Output | Content |
|---|---|
| `sample_graph_input.h5ad` | Cells passing RNA QC with at least one usable TCR chain |
| `{SAMPLE_ID}_rna_tcr_all.h5ad` | All input RNA cells with matched TCR annotations and QC flags |
| `{SAMPLE_ID}_contig_audit.csv.gz` | Contig selection audit |
| `{SAMPLE_ID}_pairing_qc.json` | Pairing and filtering summary |

After preparation, run the following commands from the `BTC_demo` directory:

```bash
python creat_graph.py outputs/prepared/sample_graph_input.h5ad --output-dir my_graphs --device cuda:0
python byster.py --graph-dir my_graphs --output-dir my_predictions --device cuda:0
```

Rerunning the example replaces outputs with the same names. Rebuild graphs whenever the input cells change.

## Issues

For questions or reproducibility problems, open an [issue](https://github.com/C8H8O-tenes/BTC/issues) and include the commit or release, environment details, command, and error message.
