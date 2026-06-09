# ICP_stage4 Multi-Field Structure-Feature Training Evaluation

## Scope

This run uses the reorganized ICP_stage4 multi-field structure dataset:

- Dataset: `data/outputs_icp_stage4_enriched_360_csv_npz_multifield_structure_v1`
- Cases: 360, from 60 coil-structure groups x 6 process conditions
- Grid: `[440, 600]`
- Scalar process conditions used by the current checkpoint/config: `pp`, `pp0`
- Geometry source variables used only to generate structures: `llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`
- Structure input: `part_lite_v1` feature maps from per-case geometry/part masks
- Targets: `ne`, `ni`, `Te`, `phi`, `Br`, `Bz`, `Jelr`, `Jelz`

The coil geometry variables are not used as raw scalar condition inputs. They are converted into structure masks/SDF/part summary features, then learned as spatial structure information together with scalar process conditions.

## Feature And Preprocessing Plan Used

The full pilot uses:

- `input_mode: table_plus_structure`
- `geometry_provider_mode_effective: parametric_parts`
- `structure_feature_profile_effective: part_lite_v1`
- Feature channels: `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any`, `normal_x`, `normal_y`, `curvature_proxy`, `boundary_band`, `part_sdf_nearest`, `part_sdf_second`, `part_gap_proxy`, `solid_proximity`
- Condition scaler: `robust`
- Density transforms: `ne`, `ni` use `log10_floor`
- Temperature transform: `Te` uses `log1p`
- Potential transform: `phi` uses `signed_log1p`
- Magnetic/current fields in v2: `Br`, `Bz`, `Jelr`, `Jelz` use `identity + robust`

The v1 pilot used `signed_log1p` for the signed magnetic/current channels. That made inverse reconstruction unstable for `Jelz`, so v2 switches signed field targets to identity scaling with robust normalization.

## Execution

Smoke:

- Command: `scripts/run_icp_stage4_core4_benchmarks.py --sizes smoke --models ffno ...generated_multifield_part_lite_v1_pilot_v2`
- Status: passed
- Time: 28.8 s

Full pilot:

- Command: `scripts/run_icp_stage4_core4_benchmarks.py --sizes full --models ffno ...generated_multifield_part_lite_v1_pilot_v2`
- Status: passed
- Epochs: 20
- Time: 2440.1 s
- Leaderboard: `runs/icp_stage4_multifield_part_lite_v1_pilot_v2/full/ffno/leaderboard.csv`

## Results

| Run | Mean plasma R2 | ne R2 | ni R2 | Te R2 | phi R2 | Br R2 | Bz R2 | Jelr R2 | Jelz R2 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 signed-current transforms | -31678.345 | -0.229 | -0.229 | -0.060 | 0.349 | 0.510 | 0.245 | 0.035 | -253427.378 | rejected |
| v2 identity signed fields | 0.158 | -0.227 | -0.229 | -0.043 | 0.378 | 0.548 | 0.264 | 0.204 | 0.367 | pipeline pilot accepted |

v2 fixed the catastrophic current-density inverse-transform failure and produced finite predictions for all targets:

- `test_finite_ratio_*_plasma = 1.0` for all eight targets.
- `ne`, `ni`, `Te` are still weak after 20 epochs and should not be used as the final density/uniformity optimization surrogate.
- `phi`, `Br`, `Bz`, `Jelr`, `Jelz` show usable first-pass signal, with `Br` strongest among the added physical fields.

Negative ratios for `Br`, `Bz`, `Jelr`, and `Jelz` are not failure indicators because these signed fields can physically be negative. Density and temperature positivity remained stable (`ne`, `ni`, `Te` negative ratios are 0).

## Interpretation

The current multi-field formulation is technically valid: process scalar conditions and structure feature maps feed an 8-field FFNO surrogate, and the reorganized dataset is trainable end-to-end. The pilot also confirms that current-density channels should not use an exponential inverse transform.

The result is not yet strong enough for final coil optimization driven by `ne` uniformity, because density R2 is still negative on the structure/extrapolation split. This is expected for a 20-epoch 8-target pilot with only 360 cases and a large `[440, 600]` field. The previous Core4-only structure-feature models remain better density surrogates until this multi-field model is trained longer and cross-checked.

## Recommended Next Training

1. Run v2 for 80 epochs with the same preprocessing.
2. Compare `part_lite_v1` against `icp_part_sdf_lite_v1` and `struct_spatial_v1` for the same eight targets.
3. Add at least one cross-check model family: `unet` or `cno_operator_unet`.
4. For `Br`/`Bz`, add valid-field or chamber-region evaluation in a future external evaluation pass. The current benchmark reports plasma-only metrics uniformly.
5. Keep optimization gated by density finite/positive checks and by surrogate agreement. Use the Core4 density surrogate for near-term density optimization, and use this multi-field pilot as a physics-field consistency probe until its density metrics improve.

