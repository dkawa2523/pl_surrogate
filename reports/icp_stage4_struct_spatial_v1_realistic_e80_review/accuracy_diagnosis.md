# ICP Stage4 Struct-Spatial v1 Accuracy Diagnosis

## Current Accuracy

| model | mean R2 | ne R2 | ni R2 | Te R2 | phi R2 |
|---|---:|---:|---:|---:|---:|
| UNet | 0.8548 | 0.8374 | 0.8375 | 0.8468 | 0.8976 |
| FFNO | 0.7770 | 0.5853 | 0.5865 | 0.9563 | 0.9799 |
| CNO | 0.7569 | 0.5700 | 0.5691 | 0.9445 | 0.9441 |

FFNO/CNO are good on `Te` and `phi`, but weak on `ne`/`ni`. The density drop is mainly in the deep plasma region, not only at boundaries:

| model | ne boundary R2 | ne deep R2 | ni boundary R2 | ni deep R2 |
|---|---:|---:|---:|---:|
| UNet | 0.6770 | 0.8339 | 0.6712 | 0.8339 |
| FFNO | 0.6250 | 0.5699 | 0.6298 | 0.5712 |
| CNO | 0.6797 | 0.5544 | 0.6773 | 0.5535 |

## Structure Input Check

- The run is using case-varying structure input:
  - `cond_dim=2`: `pp, pp0`
  - structure profile: `icp_struct_spatial_v1`
  - logical channels: `x, y, mask_plasma, distance_signed, distance_any, mask_coil, distance_coil, coil_proximity`
  - compact storage: static `[5,440,600]` + case structure `[360,3,440,600]`
- The model receives structure tensors, but the current extrap evaluation does not prove structure generalization.
- Reason: all 60 `base_case_id` structures appear in both train and test. The test is mainly a high-`pp` process extrapolation:
  - train `pp`: 505.725 to 2097.135
  - val `pp`: 2098.915 to 2511.754
  - test `pp`: 2530.878 to 2999.798
  - train-test base structure overlap: 60 / 60

## Density Difficulty

The density target distribution shifts strongly in the high-`pp` test split:

| var | split | mean | std | p99 | max |
|---|---|---:|---:|---:|---:|
| ne | train | 2.23e17 | 3.92e17 | 1.87e18 | 3.76e18 |
| ne | test | 5.71e17 | 9.65e17 | 4.56e18 | 6.25e18 |
| ni | train | 2.23e17 | 3.92e17 | 1.87e18 | 3.76e18 |
| ni | test | 5.71e17 | 9.65e17 | 4.56e18 | 6.25e18 |

`Te` and `phi` do not shift as severely, so FFNO/CNO can score well there while missing density amplitude and core distribution.

## Likely Causes

1. The current split is not a structure holdout. It verifies high-pressure extrapolation with known structures, not unknown-geometry learning.
2. The current structure representation is union-coil only. It is useful, but it cannot distinguish part identity or part-level shape changes.
3. FFNO was intentionally lightened: width 32, 3 layers, 8 modes. This favors smooth low-frequency fields and is likely too small for high-density extrapolation.
4. The current CNO run uses `cno` / `CNOBaseline`, a simple local residual convolution model. It is not the stronger multiscale `cno_operator_unet` already present in the codebase.
5. Spatial continuous features are not z-score scaled in this run. `distance_coil` remains raw up to 663 px, while masks/proximities are 0..1.
6. Scalers in preprocessing are fitted on the generic random split, not the primary high-`pp` extrap split. This is a purity issue and should be aligned, even though it does not explain the low FFNO/CNO density score by itself.
7. The loss is fixed equal-weight multitask Huber on standardized targets. Density has a much heavier tail under high `pp`, so the easy smooth fields can dominate model selection unless density-specific diagnostics and weights are made explicit.

## Recommended Next Steps

1. Add a structure-holdout evaluation split grouped by `base_case_id`. Keep the current pressure extrap split, but do not use it as proof of structure learning.
2. Align cond/y scaler fitting with the active primary split train set. Keep density linear `identity`.
3. Enable train-split z-score scaling for continuous spatial channels, while leaving masks unscaled.
4. Run a small density-focused ablation:
   - UNet current config as baseline
   - FFNO with more modes/capacity, e.g. width 48 or 64, modes 16, layers 4
   - CNO replaced by existing `cno_operator_unet`, not the simple local `cno`
5. Make checkpoint selection report per-variable R2 for `ne`, `ni`, `Te`, `phi`. Current selection uses all variables, but the CSV only surfaces `Te` and `phi`.
6. If density remains the optimization-critical target, try modest density loss weighting, e.g. `ne=1.5, ni=1.5, Te=1.0, phi=1.0`, before adding new feature contracts.
7. Defer part-level SDF until the split/scaler/capacity issues are cleanly measured. When added, keep it small and slot-based.
