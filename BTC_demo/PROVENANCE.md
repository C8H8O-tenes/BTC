# Source provenance

All changes are confined to BTC_demo. Original scripts and data are unchanged.

- `creat_graph.py`: copied from `sample/creat_graph.py`; adapted portable resource paths, default target size, count validation, device/thread selection and progress messages.
- `BTC_model.py`: exact extraction of `HeteroRelationLayer` and `HeteroFinetuner` from `database_label_finetune/finetune_core.py`.
- `byster.py`: inference adapted from `GSE243013_lung_application/BTC_GNN_lastlayer_infer.ipynb`, with a standalone CLI; uses the bundled DATABASE checkpoint, not the notebook's lung checkpoint. Internal BTC architecture and checkpoint semantics are retained.
- Example and models: copied unchanged from `sample/sample_graph_input.h5ad`, `sample/data/DATABASE/`, and `sample/data/scGPT_human/`.
- Hallmark and segment vocabulary: copied unchanged from `sample/data/`; order retained for model compatibility.
- Environment snapshots: copied unchanged from sample; preparation notebook adapted from sample/md.ipynb.
- `validation/source_manifest.json` records original sources/checksums and historical initial destination paths. The graph script is now at creat_graph.py.

The initial combined run.py entry point has been removed in favor of independent graph construction and inference. Existing validation outputs from that initial layout are retained as historical evidence.

No new license is asserted over the source project or bundled third-party assets.
