# ICP_stage4 Training Evaluation Summary

## Scope

This evaluation follows the conference problem setting:

- scalar conditions are `pp` and `pp0` only
- coil geometry source columns are not fed as scalar model inputs
- coil structure is represented through structure feature maps
- target fields are the current Core4 set: `ne`, `ni`, `Te`, `phi`

`part_lite_v1` was not run because the current audited structure pack has
`part_mask_stack_cases = 0`. It remains the intended next primary comparison
after regenerating the structure pack with per-part masks.

## Runs

| profile | model | status | runtime | role |
| --- | --- | ---: | ---: | --- |
| `icp_part_sdf_lite_v1` | FFNO | passed | 7162.2 s | current runnable optimization candidate |
| `icp_struct_spatial_v1` | FFNO | passed | 5952.8 s | current best field-prediction baseline |
| `icp_part_sdf_lite_v1` | UNet | existing reference | not rerun | cross-model reference only |

## Main Metrics

| profile | model | mean plasma R2 extrap | ne R2 | ni R2 | Te R2 | phi R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `icp_part_sdf_lite_v1` | FFNO | 0.9658 | 0.9448 | 0.9358 | 0.9890 | 0.9938 |
| `icp_struct_spatial_v1` | FFNO | 0.9878 | 0.9872 | 0.9762 | 0.9944 | 0.9934 |
| `icp_part_sdf_lite_v1` | UNet reference | 0.9010 | 0.8829 | 0.8822 | 0.9013 | 0.9376 |

| profile | model | ne RMSE plasma | ni RMSE plasma | Te RMSE plasma | phi RMSE plasma |
| --- | --- | ---: | ---: | ---: | ---: |
| `icp_part_sdf_lite_v1` | FFNO | 2.266e17 | 2.445e17 | 0.1471 | 0.3107 |
| `icp_struct_spatial_v1` | FFNO | 1.091e17 | 1.487e17 | 0.1044 | 0.3212 |
| `icp_part_sdf_lite_v1` | UNet reference | 3.302e17 | 3.311e17 | 0.4402 | 0.9861 |

## Physical Validity Checks

| profile | model | ne negative ratio | ni negative ratio | finite ratio |
| --- | --- | ---: | ---: | ---: |
| `icp_part_sdf_lite_v1` | FFNO | 0.0 | 0.0 | 1.0 for all Core4 targets |
| `icp_struct_spatial_v1` | FFNO | 0.0 | 0.0 | 1.0 for all Core4 targets |
| `icp_part_sdf_lite_v1` | UNet reference | 0.00577 | 0.00763 | not rechecked in this run |

## Interpretation

For pure field prediction on the current Core4 dataset, `icp_struct_spatial_v1`
is the stronger baseline. It improves the mean plasma extrapolation R2 from
0.9658 to 0.9878 and roughly halves the electron-density RMSE relative to the
current `icp_part_sdf_lite_v1` FFNO run.

For downstream geometry optimization, `icp_part_sdf_lite_v1` remains important
because it uses the `parametric_parts` provider path. That makes it the safer
current candidate when optimization candidates are generated from coil geometry
and converted back into structure features. The `icp_struct_spatial_v1` run is
best treated as the current field-accuracy baseline unless its feature
regeneration path is made equally parametric.

The existing UNet reference underperforms FFNO on this dataset and has nonzero
negative density ratios, so it should be used only as a cross-check, not as the
primary optimization surrogate.

## Conference-Level Conclusion

The current results support the problem formulation: structure-feature inputs
can learn the Core4 two-dimensional plasma fields with high held-out-structure
accuracy while keeping raw coil dimensions out of the scalar condition vector.

The strongest immediate result is:

- `icp_struct_spatial_v1 + FFNO` for field-prediction accuracy

The most practical immediate optimization surrogate is:

- `icp_part_sdf_lite_v1 + FFNO` because it is compatible with the parametric
  parts geometry path

The next required experiment is:

- regenerate the structure pack with `part_mask_stack`
- run `part_lite_v1 + FFNO/UNet/CNO`
- compare order-invariant part summary features against both current baselines

## Artifacts

- summary CSV: `reports/icp_stage4_conference_problem_setting/training_evaluation_summary.csv`
- part SDF leaderboard: `runs/icp_stage4_conference_part_sdf_lite_v1_e80_baseline/full/ffno/leaderboard.csv`
- spatial leaderboard: `runs/icp_stage4_conference_struct_spatial_v1_e80_baseline/full/ffno/leaderboard.csv`
- generated summaries:
  - `reports/icp_stage4_conference_problem_setting/summary_part_sdf_lite_v1/comparison_summary.csv`
  - `reports/icp_stage4_conference_problem_setting/summary_struct_spatial_v1/comparison_summary.csv`
