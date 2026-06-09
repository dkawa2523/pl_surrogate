# ICP Stage4 Spatial Distribution Improvement Plan

## Problem

The current R2 values do not fully reflect visual field quality. A model can keep acceptable pixel-wise R2 while the predicted spatial distribution looks wrong: peaks are too smooth, core density is shifted or flattened, boundary gradients are weak, or total plasma inventory is biased.

This is especially visible for density. In the current high-`pp` extrapolation split, `ne`/`ni` distribution shifts strongly from train to test, while `Te`/`phi` are easier. FFNO/CNO therefore learn smooth fields well but miss density amplitude and structure.

## Why R2 Hides The Issue

1. Pixel-wise R2 is dominated by high-variance regions and does not measure topology, peak location, integral, or profile shape.
2. The loss is equal-weight multitask Huber on standardized targets. Large density peak errors enter the Huber linear-gradient regime, so the model is not strongly pushed to fix high-density peaks.
3. `spatial_consistency` is currently disabled, so blurry but numerically acceptable fields are not penalized for weak gradients or lost localized structure.
4. Density is learned as a direct field. This entangles global amplitude growth with spatial shape learning, which is difficult under high-`pp` extrapolation.
5. Current structure input is union-coil only. It is valid for coarse structure awareness, but not enough for part-level geometry sensitivity or optimization.
6. Spatial continuous features are not scaled in the current run, while `distance_coil` can be hundreds of pixels. This is a conditioning issue for FFNO/CNO.
7. The current `extrap` split is a pressure extrapolation with known structures, not a structure holdout. It cannot validate geometry generalization.

## Design Direction

Keep density in linear physical units. Do not reintroduce log-density training.

The main correction is to separate three concerns:

1. **Evaluation**: add spatial-distribution metrics beyond R2.
2. **Learning target**: make density amplitude and density shape easier to learn without changing the final physical output.
3. **Geometry sensitivity**: validate structure learning with a structure-holdout split before adding more geometry channels.

## Concrete Changes

### 1. Add Distribution Metrics

Add the following to evaluation reports and plots:

- `mass_rel_error`: relative error of plasma integral for each variable.
- `peak_rel_error`: relative error of max / high quantile, especially p95/p99 density.
- `peak_location_error_px`: distance between true and predicted peak locations.
- `center_of_mass_error_px`: density-weighted centroid error.
- `profile_rmse_r`, `profile_rmse_z`: line/profile error after averaging over each axis.
- `grad_rmse`: gradient-field RMSE in plasma.
- `top_quantile_rmse`: RMSE on top 10% density pixels.
- `shape_corr`: correlation after normalizing each case by its plasma mean.
- `ne_ni_consistency`: RMSE or relative difference between `ne` and `ni`.

These metrics should be by-case and aggregated by split, with plots for worst cases. This makes "looks wrong" measurable.

### 2. Fix Evaluation Foundation

- Fit cond/y scalers using the active primary split train indices, not a generic random split.
- Add separate protocol splits:
  - `pressure_extrap`: current high-`pp` extrapolation.
  - `structure_holdout`: group by `base_case_id`, unseen structures in test.
  - `structure_pressure_holdout`: unseen structures plus high-`pp`, used only after the first two are stable.
- Keep current `icp_struct_spatial_v1` as the baseline structure representation.

### 3. Improve Input Conditioning

- Enable train-split scaling for continuous spatial channels:
  - scale: `x`, `y`, `distance_signed`, `distance_any`, `distance_coil`
  - keep unscaled: `mask_plasma`, `mask_coil`
  - `coil_proximity` can stay 0..1 or be z-scored only if metrics show benefit.
- Keep compact pack design: static `[5,H,W]` + case `[N,3,H,W]`.

### 4. Density Amplitude-Shape Learning

Introduce a small density target decomposition internally:

```text
density_linear = density_amplitude * density_shape
```

where:

- `density_amplitude`: per-case scalar such as plasma mean or integral.
- `density_shape`: linear density divided by amplitude, normalized around order 1.
- final prediction is reconstructed back to linear `ne` / `ni`.

This is not log learning. It keeps final density linear while making high-`pp` amplitude extrapolation and spatial shape learning separable.

Initial implementation should be minimal:

- train scalar heads for `ne_amp`, `ni_amp`
- train spatial heads for `ne_shape`, `ni_shape`
- reconstruct `ne`, `ni` for loss/eval
- keep existing direct `Te`, `phi`
- report both amplitude error and shape error

### 5. Add Shape-Aware Loss Terms

Use small, targeted terms:

- base supervised Huber remains.
- density top-quantile loss for `ne`/`ni`, applied only inside plasma.
- low-weight gradient consistency for density shape, not all variables.
- integral consistency for `ne`/`ni` over plasma.

Recommended first weights:

```yaml
density_shape:
  top_quantile_weight: 0.10
  grad_weight: 0.02
  integral_weight: 0.05
```

Avoid large physics contracts here. These are direct field-quality terms.

### 6. Model Candidates

Use UNet as the reliable baseline, then test operator models under the corrected setup.

- UNet:
  - keep current model as baseline.
  - consider `output_heads.mode: split_density_field`.
- FFNO:
  - current `width=32, layers=3, modes=8` is too light for density.
  - try `width=48 or 64`, `layers=4`, `modes=16`, local skip enabled with nonzero init.
- CNO:
  - current `cno` is a simple local Conv baseline.
  - use existing `cno_operator_unet` for real multiscale field learning.

### 7. Geometry Features

Do not add a large part-SDF contract yet.

First prove:

- pressure extrap improves with density decomposition and shape metrics.
- structure_holdout is measurable and stable with union-coil features.

Then add a minimal part-aware feature set:

- fixed K coil slots
- `sdf_coil_01..K` only, no descriptor/latent path initially
- use it only if structure_holdout shows union-coil ambiguity.

## Execution Order

1. Add distribution metrics and worst-case plots.
2. Fix scaler fit to active primary split.
3. Add `structure_holdout` split.
4. Enable continuous spatial feature scaling.
5. Run short UNet/FFNO/CNO-operator probes.
6. Add density amplitude-shape mode if the distribution metrics confirm peak/integral errors.
7. Run 80epoch only for the best two candidates.

## Expected Outcome

The goal is not only higher R2. The accepted model should satisfy:

- density R2 improves,
- density integral error decreases,
- peak error decreases,
- mean profiles align,
- worst-case plots no longer show collapsed or shifted structures,
- structure_holdout does not collapse relative to pressure-only extrapolation.
