# ICP Stage4 Struct-Spatial v1 Realistic E80 Review

- Dataset: `data/outputs_icp_stage4_enriched_360_csv_npz_core4_struct_spatial_v1`
- Structure input: case-varying 2D grid tensor, `icp_struct_spatial_v1`
- Condition vector: process-only `pp, pp0` (`cond_dim=2`)
- Density transform: `identity`; plots are linear physical units, not log scale.

## What Changed

- Distance-field preprocessing was changed from Python pixel-loop/BFS to a vectorized Manhattan distance pass. This keeps the same distance definition but removes the major preprocessing stall.
- Long grid training now writes:
  - `scalars/metrics_partial.csv`
  - `scalars/progress_latest.json`
  - `scalars/optimization_diagnostics_partial.csv`
- Realistic FFNO/CNO configs keep the same 8 logical structure channels but reduce model/batch pressure:
  - FFNO: batch 4, width 32, 3 layers, 8 modes
  - CNO: batch 8, width 32, 3 layers

## Accuracy

| model | R2 mean | ne | ni | Te | phi | selected epoch | train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| unet | 0.8548 | 0.8374 | 0.8375 | 0.8468 | 0.8976 | 70 | about 1h |
| ffno | 0.7770 | 0.5853 | 0.5865 | 0.9563 | 0.9799 | 60 | 3432 |
| cno | 0.7569 | 0.5700 | 0.5691 | 0.9445 | 0.9441 | 75 | 3096 |

UNet remains the best baseline for this dataset. FFNO/CNO learn `Te` and `phi` very well, but their density accuracy is much weaker. That means they are not yet the right primary surrogate if the optimization objective depends strongly on density structure.

## Cost

- Full 80epoch FFNO and CNO now run in roughly one hour each instead of saturating VRAM indefinitely.
- VRAM during the realistic runs was about 4.4GB for FFNO and 6.5GB for CNO.
- The earlier full FFNO failure was not caused by the structure concept itself. It was caused by a too-heavy operator setting plus slow preprocessing and no epoch-level progress artifacts.

## Spatial Outputs

- `plots/learning_curve_compare.png`
- `plots/learning_curve_unet.png`
- `plots/learning_curve_ffno.png`
- `plots/learning_curve_cno.png`
- `plots/r2_compare.png`
- `plots/spatial_region_r2_unet.png`
- `plots/spatial_region_r2_ffno.png`
- `plots/spatial_region_r2_cno.png`
- `plots/spatial_region_compare_all_plasma.png`
- `plots/spatial_region_compare_boundary_in.png`
- `plots/spatial_region_compare_plasma_deep.png`
- `plots/spatial_region_rmse_ne.png`
- `plots/spatial_region_rmse_ni.png`
- `plots/spatial_density_mean_compare_ne_linear.png`
- `plots/spatial_density_mean_compare_ni_linear.png`
- `plots/spatial_fields_extrap_mean_72cases_unet.png`
- `plots/spatial_fields_extrap_mean_72cases_ffno.png`
- `plots/spatial_fields_extrap_mean_72cases_cno.png`
- `plots/spatial_fields_extrap_case_case_g008_op01_unet.png`
- `plots/spatial_fields_extrap_case_case_g008_op01_ffno.png`
- `plots/spatial_fields_extrap_case_case_g008_op01_cno.png`

## Optimization Readiness

- Current input is suitable for structure-aware surrogate learning at the union-coil level: `mask_coil`, `distance_coil`, and `coil_proximity` are explicit spatial fields.
- For immediate black-box structure search, use the UNet checkpoint as the primary model because density performance is materially better.
- For part-level coil shape optimization, the remaining limitation is not the training loop. It is representation: union `mask_coil` does not preserve which part changed.
- The next useful feature step should be small: add a bounded number of part-level SDF/soft-mask channels only for controllable coil slots. Avoid a broad descriptor/latent contract until a concrete optimization objective requires it.

## Recommendation

Keep `icp_struct_spatial_v1` as the current production structure input. Use UNet as the optimization baseline. Treat FFNO/CNO as secondary field-quality comparisons until their density branch is improved.
