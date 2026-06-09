# ICP Stage4 Struct-Spatial v1 E80 Review

- Run root: `runs/icp_stage4_core4/struct_spatial_v1_e80_extrap_primary_batched_eval`
- Primary split: `extrap`; test cases: `72`
- Best primary model: `unet` (`R2 extrap mean=0.7492`)

## Metrics

| model | hours | R2 interp | R2 extrap | R2 dual | ne | ni | Te | phi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unet | 10.46 | 0.4242 | 0.7492 | 0.5867 | 0.6145 | 0.6398 | 0.8414 | 0.9011 |
| ffno | 26.26 | 0.3444 | 0.6442 | 0.4943 | 0.5071 | 0.5500 | 0.6555 | 0.8641 |
| cno | 18.81 | 0.8567 | 0.6647 | 0.7607 | 0.4969 | 0.4671 | 0.7989 | 0.8958 |

## Plots

- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/r2_summary.png`
- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/runtime_hours.png`
- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/spatial_abs_error_extrap_mean_72cases.png`
- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/spatial_fields_extrap_case_case_g008_op01.png`
- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/spatial_fields_extrap_mean_72cases.png`
- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/spatial_density_linear_case.png`
- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/spatial_density_abs_error_linear_case.png`
- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/spatial_density_linear_mean.png`
- `reports/icp_stage4_struct_spatial_v1_e80_batched_eval_review/plots/spatial_density_abs_error_linear_mean.png`

## Notes

- These maps are generated directly from the extrap checkpoints because case-varying structure inputs skip the older inference/single artifact path.
- Structure is provided as a case-varying 2D grid tensor, not as scalar coil parameters in the condition vector. The pack shape is `[360, 8, 440, 600]` with channels `x, y, mask_plasma, distance_signed, distance_any, mask_coil, distance_coil, coil_proximity`.
- Density plots are shown in linear physical units. Color limits use p99.5 clipping for readability.
- The generated struct-spatial benchmark configs have been corrected so `ne` and `ni` preprocessing uses `value_transform: identity`; rerunning is required for new checkpoints trained with that corrected setting.

## Runtime Notes

- Full run elapsed time was about 55.5 hours: UNet 10.5h, FFNO 26.3h, CNO 18.8h.
- Each full model uses dual-axis evaluation, so 80 epochs are run twice per model: `interp` and `extrap`.
- Each case is a 440x600 grid with 8 case-varying spatial structure channels plus 2 process conditions. With `batch_size_cases=12`, one training step processes about 3.17M grid points.
- The case spatial pack shape is `[360, 8, 440, 600]`, so preprocessing and each training batch move large tensors through GPU memory.
- FFNO/CNO are heavier than UNet here because they run operator-style grid layers on the full 440x600 field; FFNO was the slowest in this run.
- The older inference/single path is skipped for case-varying structure, so spatial plots are regenerated directly from checkpoints after training.
