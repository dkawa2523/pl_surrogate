# ICP Stage4 Experiment Archive

This folder keeps ICP Stage4 experiment-only utilities outside the product
mainline. They are useful for reproducing historical structure-feature studies,
but they are not imported by the core package or default tests.

Scripts in `scripts/` may keep fixed dataset roots, publication plotting
choices, or one-off optimization assumptions. Prefer the main `plasma_surrogate`
CLI and maintained `configs/` entry points for product workflows.

Example historical entry points:

```powershell
py experiments\icp_stage4\scripts\run_icp_stage4_core4_benchmarks.py
py experiments\icp_stage4\scripts\run_icp_part_sdf_shape_optimize.py
```
