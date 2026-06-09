# ICP Stage4 Struct Spatial v1 Distribution Probe Review

## Run Scope

- Dataset/config: `generated_struct_spatial_v1_distribution_v2_probe_e20_primary`
- Split: `extrap`
- Scaler fit split: `extrap`
- Density transform: `identity` for `ne` and `ni`
- Structure input: case-varying 2D structure features enabled
- Compact spatial pack: enabled

## 20epoch Probe Result

| model | status | time | ne R2 | ni R2 | Te R2 | phi R2 | ne dist score | ni dist score | ne integral err | ni integral err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNet | passed | 25.4 min | 0.8759 | 0.8731 | 0.9126 | 0.9319 | 0.2455 | 0.2842 | 0.2099 | 0.2173 |
| FFNO | passed | 32.4 min | 0.9506 | 0.9514 | 0.9344 | 0.9699 | 0.1384 | 0.1382 | 0.1217 | 0.1350 |
| CNO operator UNet | stopped_slow | 20.9 min | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Interpretation

- FFNO is the current best 80epoch candidate. It improved both density R2 and distribution quality versus UNet.
- The earlier FFNO density collapse was not reproduced under the v2 setup. The fixes to scaler fit split, structure inputs, and density/distribution evaluation materially changed the result.
- CNO operator UNet is not currently a realistic candidate at this resolution/config. It used nearly all 16GB VRAM and did not complete the first epoch after about 21 minutes.
- Distribution metrics still show non-trivial density amplitude/shape error even for FFNO. FFNO is better, but not “solved”: integral error remains around 12-14% and p99 relative error around 15%.

## Artifacts

- Summary CSV: `reports/icp_stage4_struct_spatial_v1_distribution_v2_probe_e20_review/comparison_summary.csv`
- Accuracy plot: `reports/icp_stage4_struct_spatial_v1_distribution_v2_probe_e20_review/plots/accuracy_r2_comparison.png`
- Distribution plot: `reports/icp_stage4_struct_spatial_v1_distribution_v2_probe_e20_review/plots/distribution_metrics_comparison.png`
- Learning curves: `reports/icp_stage4_struct_spatial_v1_distribution_v2_probe_e20_review/plots/learning_curves_comparison.png`
- UNet worst maps: `runs/icp_stage4_struct_spatial_v1_distribution_v2_probe_e20_primary/full/unet/models/unet/eval_protocol/extrap/eval/plots/`
- FFNO worst maps: `runs/icp_stage4_struct_spatial_v1_distribution_v2_probe_e20_primary/full/ffno/models/ffno/eval_protocol/extrap/eval/plots/`

## Next Step

Run FFNO 80epoch first. Keep UNet as a secondary baseline if time allows. Do not run the current CNO operator UNet full setting again until it is downscaled or redesigned for this grid size.
