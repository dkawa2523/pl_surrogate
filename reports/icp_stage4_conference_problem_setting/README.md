# ICP_stage4 Conference Problem Setting Run Notes

This folder records the read-only dataset audit used to turn the conference
problem setting into runnable study settings.

## Current Result

- Audit output: `dataset_audit_part_lite_ready.json`
- Rows: 360
- Structure groups: 60
- Cases per structure: 6
- Recommended scalar conditions: `pp`, `pp0`
- Geometry source columns: `llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`
- Current structure pack has fixed-slot SDF channels.
- Current structure pack does not contain `part_mask_stack`.

## Multi-Field Dataset

The unified problem-setting dataset has now been organized at:

- `data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1`

It contains:

- scalar process conditions available in the source data: `pp`, `pp0`
- geometry source columns for feature generation only: `llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`
- structure features: masks, `part_mask_stack`, and `sdf_coil_01` ... `sdf_coil_06`
- 2D target fields: `ne`, `ni`, `phi`, `Te`, `Br`, `Bz`, `Jelr`, `Jelz`

COMSOL complex phasor values in electromagnetic/current fields are converted
with `complex_field_mode=magnitude`.

## Consequence

The current runnable primary setting is:

- `configs/experimental/icp_stage4/generated_conference_part_sdf_lite_v1_e80_baseline`

The spatial baseline is:

- `configs/experimental/icp_stage4/generated_conference_struct_spatial_v1_e80_baseline`

The intended order-invariant primary setting was generated but should not be
used until the dataset is regenerated with `part_mask_stack`:

- `configs/experimental/icp_stage4/generated_conference_part_lite_v1_e80`

## Goal Adjustment For Presentation

The conference claim should be framed as:

1. Reformulate ICP surrogate learning from dimension-vector regression to
   structure-feature 2D field prediction.
2. Demonstrate the current runnable structure-feature baselines on Core4 fields.
3. Treat `part_lite_v1` as the next primary comparison after regenerating the
   structure pack with per-part masks.
4. Treat optimization results as candidate screening, not final coil design,
   until cross-surrogate or COMSOL re-evaluation is completed.

## Smoke Runs

Two 1-epoch smoke runs were executed to verify that the runnable settings are
wired correctly. These are not accuracy results for the conference.

| profile | model | status | seconds | leaderboard |
| --- | --- | ---: | ---: | --- |
| `icp_part_sdf_lite_v1` | FFNO | passed | 20.8 | `runs/icp_stage4_conference_part_sdf_lite_v1_e80_baseline/smoke/ffno/leaderboard.csv` |
| `icp_struct_spatial_v1` | FFNO | passed | 17.2 | `runs/icp_stage4_conference_struct_spatial_v1_e80_baseline/smoke/ffno/leaderboard.csv` |

The smoke runs confirm finite training/evaluation, structure-feature runtime
selection, robust/log target preprocessing, Huber supervised loss, and the
generated inference/optimization blocks.

## Full Training Evaluation

Full FFNO evaluations were run for the two currently executable conference
settings:

| profile | model | status | seconds | mean plasma R2 extrap | role |
| --- | --- | ---: | ---: | ---: | --- |
| `icp_part_sdf_lite_v1` | FFNO | passed | 7162.2 | 0.9658 | current runnable optimization candidate |
| `icp_struct_spatial_v1` | FFNO | passed | 5952.8 | 0.9878 | current best field-prediction baseline |

Detailed evaluation:

- `training_evaluation_summary.md`
- `training_evaluation_summary.csv`
