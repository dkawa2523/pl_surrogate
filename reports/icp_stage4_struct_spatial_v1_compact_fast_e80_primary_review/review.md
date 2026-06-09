# ICP Stage4 Compact Struct-Spatial v1 Review

- Run root: `runs/icp_stage4_core4/struct_spatial_v1_compact_fast_e80_primary`
- Completed full run: `unet`, 80 epochs, primary split `extrap`
- Smoke runs completed: `unet`, `ffno`, `cno`
- `ffno` full 80epoch was stopped because it saturated GPU memory and produced no final metrics after the preprocessing stage.

## Input Check

- Condition vector is process-only: `pp, pp0` (`cond_dim=2`).
- Structure input is case-varying 2D grid features, not scalar coil parameters.
- Logical channels remain 8:
  `x, y, mask_plasma, distance_signed, distance_any, mask_coil, distance_coil, coil_proximity`.
- Compact storage is active:
  - static pack: `[5, 440, 600]`
  - case pack: `[360, 3, 440, 600]`
  - logical batch input: `[B, 440, 600, 8]`
- Density target transforms are `identity`; the generated spatial plots are linear physical-unit plots, not log-scale plots.

## UNet 80epoch Metrics

| metric | value |
|---|---:|
| primary `test_r2_plasma_mean_extrap` | 0.8548 |
| `ne` R2 plasma extrap | 0.8374 |
| `ni` R2 plasma extrap | 0.8375 |
| `Te` R2 plasma extrap | 0.8468 |
| `phi` R2 plasma extrap | 0.8976 |

Training ran 80 epochs. Loss moved from train/val `0.8516/0.8545` to `0.0823/0.1193`; selected epoch was 70.

## Spatial Summary

All-plasma RMSE/R2:

| var | RMSE | R2 |
|---|---:|---:|
| `ne` | `3.8903e17` | 0.8374 |
| `ni` | `3.8885e17` | 0.8375 |
| `Te` | 0.5483 | 0.8468 |
| `phi` | 1.2628 | 0.8976 |

Region R2 shows density is still weaker near the plasma boundary:

| var | boundary_in | plasma_mid | plasma_deep |
|---|---:|---:|---:|
| `ne` | 0.6770 | 0.7354 | 0.8339 |
| `ni` | 0.6712 | 0.7496 | 0.8339 |
| `Te` | 0.8550 | 0.8671 | 0.8440 |
| `phi` | 0.8626 | 0.8313 | 0.8697 |

Plots:

- `plots/r2_summary.png`
- `plots/spatial_fields_extrap_case_case_g008_op01.png`
- `plots/spatial_fields_extrap_mean_72cases.png`
- `plots/spatial_density_case_ne_linear.png`
- `plots/spatial_density_case_ni_linear.png`
- `plots/spatial_density_mean_ne_linear.png`
- `plots/spatial_density_mean_ni_linear.png`

## Cost Review

- The compact pack fixed the main avoidable memory issue: fixed `x/y/mask/distance` channels are no longer duplicated for all 360 cases.
- The earlier slow run was dominated by the NumPy `spatial_consistency.grad_huber` path; the fast configs disable it.
- UNet full 80epoch now completes in about one hour on the current GPU.
- FFNO full-resolution 80epoch is still not cost-effective in this setup: it held about 15.9GB/16.3GB VRAM and 100 percent GPU use without reaching a final metric in a reasonable time.

## Recommendation

- Treat `unet` compact struct-spatial v1 as the current baseline.
- Do not run FFNO/CNO 80epoch at full `440x600`, batch 12 as the next default.
- If FFNO/CNO are still needed, run a smaller candidate first: fewer epochs, smaller width/modes, or reduced grid/batch, then promote only if the smoke and short-run metrics beat UNet.
- Keep density learning and visualization linear; do not reintroduce log-density targets or log-density plots.
- Keep part-wise SDF out of the current path until the lighter baseline is stable and the shape-optimization objective is ready.
