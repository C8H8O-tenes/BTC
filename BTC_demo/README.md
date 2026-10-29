# BTC demo：先建图，再预测

在已经配置好的 BTC 环境中，分两步处理匹配的单细胞 RNA 原始计数和 TCR 注释。两个脚本独立运行，预测阶段可以重复使用已经构建的图。

## 1. creat_graph.py：构建图

```bash
cd BTC_demo
python creat_graph.py examples/sample_graph_input.h5ad \
  --output-dir graphs \
  --target-size 50 \
  --device cuda:0
```

执行 scGPT embedding → SEACells → Hallmark scoring → SCEPTR → 异质图构建，输出 `graphs/graph_manifest.csv`、图文件、MetaCell/MetaTCR 数据与细胞映射。此步骤不执行 BTC 分类。本机完整示例建图约需 8 分钟。

## 2. byster.py：使用 BTC 模型预测

```bash
python byster.py \
  --graph-dir graphs \
  --output-dir predictions \
  --device cuda:0
```

脚本文件使用 `byster.py` 名称；内部仍使用原 BTC 的 `HeteroRelationLayer`、`HeteroFinetuner` 架构和 checkpoint 格式。`BTC_model.py` 中的两个模型类直接提取自原 `database_label_finetune/finetune_core.py`，未改变计算逻辑。

默认权重为 `models/btc/best_finetuned_model.pt`，也可指定 `--checkpoint /path/to/best_finetuned_model.pt`。模型直接预测 MetaTCR，并通过细胞映射输出单细胞结果；此脚本不会重新建图。

两个脚本均可用 `--device cpu`；省略设备时自动选择可用的 CUDA 或 CPU。指定的输出目录必须不存在或为空。需要再次预测时换一个 `--output-dir`，无需重建图。资源相对于脚本所在目录定位，输入/输出的相对路径相对于当前工作目录解析。

## 使用自己的数据

```bash
python creat_graph.py /path/to/my_data.h5ad --output-dir my_graphs --device cuda:0
python byster.py --graph-dir my_graphs --output-dir my_predictions --device cuda:0
```

输入为 `.h5ad`，每行 RNA 与 TCR 对应同一细胞：

| 位置 | 内容 |
|---|---|
| `layers['counts']` 或 `X` | 非负原始整数 RNA 计数；默认优先 counts layer |
| `var_names` | 唯一基因符号，如 CD3D、TRAC |
| `obs_names` | 唯一细胞 ID |
| `obs['sample_id']` | 样本 ID，每个样本对应一个患者 |
| `obs['patient_id']` | 患者 ID |
| `cdr3_aa_a`、`cdr3_aa_b` | TRA/TRB CDR3 氨基酸序列，至少一条链存在 |
| `v_gene_a`, `j_gene_a`, `v_gene_b`, `d_gene_b`, `j_gene_b` | 可选 V/D/J 注释，存在时保留 |

提前完成适合自己数据的细胞质量筛选。`--tcr-mode any` 保留至少一条链的细胞；`--tcr-mode paired` 仅保留双链细胞。被排除细胞不会获得预测。不需要实验标签，不自动筛选细胞类型。

建图支持 `--sample-col`、`--patient-col`、`--counts-layer`、`--batch-size`、`--threads`、`--target-size`、`--knn`、`--seed` 等参数。默认 target_size=50、knn=10、seed=42。可用 `python creat_graph.py --help` 和 `python byster.py --help` 查看完整参数。

Hallmark 标准化使用本次输入的均值和标准差，更换输入细胞集合可能改变预测；比较结果时应保持输入与参数一致。

## 预测输出

| 文件 | 内容 |
|---|---|
| `metatcr_predictions.csv` | MetaTCR ID、TCR 注释、logit、probability、predicted_label |
| `cell_predictions.csv` | 细胞 ID、样本/患者 ID、MetaTCR ID、对应预测 |
| `qc_summary.json` | 图、MetaTCR、预测细胞数、阈值、模型信息和执行状态 |
| `run_config.json` | 预测参数与路径 |
| `error.log` | 仅在预测失败时生成 |

`probability` 是 BTC 输出的 tumor-specific 概率。默认 `>=0.5` 为 `tumor-specific`，否则为 `bystander`；可用 `--threshold` 调整。同一 MetaTCR 的细胞继承相同概率，并非逐细胞独立预测。

## 环境、示例和文件

- `BTC_environment.yml`、`requirements.txt`：原 BTC 环境快照，保留原名称和依赖。
- `creat_graph.py`：独立建图脚本，来源于原 sample 脚本。
- `byster.py`：独立预测脚本，参考原 BTC 推理 Notebook。
- `BTC_model.py`：原 BTC 模型类。
- `models/`、`resources/`、`examples/`：本地模型、基因集、词表和示例数据。
- `notebooks/quickstart.ipynb`：分开执行建图与预测。
- `notebooks/prepare_10x.ipynb`：可选的原始 RNA/TCR 转换示例。

随包 BTC 权重来自原 `sample/data/DATABASE/best_finetuned_model.pt`，为 last-layer 模型（epoch 6）。示例为 MSK1263_R1，包含 6,982 个细胞；该患者出现在 checkpoint 的训练患者名单中，示例用于验证流程，不用于独立性能评估。SCEPTR 权重来自已安装的 SCEPTR 包。

## 分发与验证

```bash
python scripts/package.py
```

生成包含代码、模型、示例和必要资源的 `dist/BTC_demo_full.zip`，不包含运行结果。解压后仍按上面两步运行。当前没有发布远程下载地址。

资源校验值见 `resources/assets.json`，原文件来源及校验值见 `validation/source_manifest.json`。详细验证见 `validation/REPORT.md`。本目录未替作者指定开源许可证；来源记录见 `PROVENANCE.md`。
