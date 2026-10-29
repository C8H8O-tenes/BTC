# Two-step entry point revision

- Public workflow: `creat_graph.py` followed by `byster.py`. Combined run.py removed.
- BTC environment snapshots unchanged; model classes in BTC_model.py match original training source exactly.
- Standalone graph CLI help and input-contract tests passed.
- Graph implementation matches the previously validated complete-run implementation, except for resolving resources from its new root-directory location. No graph computation was changed or rerun for this layout-only revision.
- Standalone `byster.py` ran successfully on the full example graphs on CPU: 3,027 MetaTCRs and 6,982 cells. All cell labels match the previous full run.
- Notebook cells and instructions now expose the two steps separately.

## Earlier full-pipeline validation


- Full example rebuilt from H5AD on CUDA 0: PASS.
- Full archive extracted into another directory and independently rebuilt on CUDA 1: PASS.
- Both runs: 6,982 retained/predicted cells, 140 MetaCells, 3,027 MetaTCRs, one graph; SEACells converged.
- Both runs: 1,907 cells classified tumor-specific at threshold 0.5.
- Relocated repeat: 0 label mismatches; maximum probability difference 1.1920929e-07.
- CPU inference on newly generated graphs: PASS; cell mapping, threshold and coverage verified.
- CPU/GPU classification parity: identical labels; max probability difference 1.7881393432617188e-07.
- Invalid graph feature dimensions rejected: PASS.
- Input tests: single-chain retention, paired mode, missing TCR exclusion, integer counts, required patient ID, and sample/patient consistency: PASS.
- CPU preflight from /tmp, both --demo and --input paths: PASS. Full CPU graph construction was not benchmarked.
- Source SHA256 checks: original copied inputs/scripts unchanged.
- Model class AST matches original training implementation exactly.
- Python and notebook cell syntax checks: PASS.
- Archive CRC check: PASS before relocated execution.

The final source additionally includes stage-progress messages and corrected optional preparation-notebook text; these do not change numerical computation. Model weights and graph parameters are unchanged.

SCEPTR emitted junction-standardization warnings for several supplied sequences; warnings are preserved in logs. Input sequences were not rewritten. This demo is a workflow validation, not an independent performance evaluation: MSK1263 appears in checkpoint training metadata.

Logs are retained locally in validation/*.log and the successful primary results in outputs/. They are excluded from the distribution archive. validation/results.json records the two run summaries.
