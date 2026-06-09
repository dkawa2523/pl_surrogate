# ICP Stage4 part SDF lite v1 review

## Setup
- Dataset: `data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1`
- Feature profile: `icp_part_sdf_lite_v1`
- Inputs: `pp, pp0` condition vector + 14 spatial channels (`x/y`, plasma/distance/coil union, `sdf_coil_01..06`)
- Split: `extrap`
- Scaler fit split: `extrap`
- Density transform: `identity`
- Models: `ffno`, `unet`
- CNO: skipped by plan because current full-resolution setting is too slow

## Training
- Smoke passed for FFNO and UNet.
- FFNO 80epoch: passed, 8506.9 sec, best validation epoch 61.
- UNet 80epoch: passed, 6848.4 sec, best validation epoch 60.

## Accuracy Summary
| model | ne R2 | ni R2 | ne distribution score | ni distribution score |
|---|---:|---:|---:|---:|
| FFNO | 0.9557 | 0.9566 | 0.3438 | 0.3189 |
| UNet | 0.8829 | 0.8822 | 0.2338 | 0.2207 |

Selection rule was density distribution quality first, R2 second. By that rule, UNet is the current best model. FFNO has clearly better R2, but its density integral/peak distribution errors are worse.

## Optimization
- Best model used: UNet
- Backend: random
- Trials: 80
- Fixed condition: `case_g002_op01`, `pp=2823.804`, `pp0=0.08349`
- Geometry space: all six coils, `tx/ty=[-0.02,0.02]`, `scale_x/scale_y=[0.9,1.1]`
- Objective: minimize `ne` uniformity
- Invalid trials: 0
- Base uniformity: 0.7875
- Best uniformity: 0.7731
- Relative improvement: 1.82%

The reference `pp` is outside the extrap train range, so this optimization is a high-pressure OOD probe. The shape parameters do affect inference and the pipeline is now usable, but the improvement is modest.

## Outputs
- `comparison_summary.csv`
- `ffno_spatial_distribution_summary.csv`
- `unet_spatial_distribution_summary.csv`
- `ffno_spatial_distribution_by_case.csv`
- `unet_spatial_distribution_by_case.csv`
- `plots/r2_comparison.png`
- `plots/density_distribution_score.png`
- `plots/ffno_learning_curve.png`
- `plots/unet_learning_curve.png`
- `plots/ffno_spatial_distribution_worst_ne_case_g027_op03.png`
- `plots/unet_spatial_distribution_worst_ne_case_g027_op03.png`
- `optimize/summary.json`
- `optimize/trials.csv`
- `plots/optimize_trace.png`
- `plots/optimize_base_vs_best_ne.png`
- `plots/optimize_base_vs_best_ni.png`

## Notes
- The part-SDF-lite path is active: compact pack is fixed 5ch + case-variable 9ch, with logical 14ch.
- `mask_plasma` is preserved during parametric optimization; coil union and six SDF channels are regenerated from `geom_param`.
- `uniformity()` was changed to compute in float64. This avoids overflow for linear density values around `1e18` and keeps optimization finite.
- Full acceptance target is only partially met: FFNO reaches density R2 >= 0.94, but distribution score is above the 0.16 target; UNet has better distribution score but lower density R2.
