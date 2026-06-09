# ICP_stage4 optimization evaluation

## Problem
- Surrogate: FFNO plasma 4-field model.
- Input: process scalars `pp`, `pp0` plus coil geometry converted to structure features.
- Search: `two_stage`, 512 trials, coil-series geometry and process variables.
- Objective: minimize `line CV(ne) / (line max ne / base line max ne)`.
- Line: row `119` (`z=5.975`), columns `0..400` (`r=0.025..20.02`), 401 samples.

## Best candidate
- Objective: `0.231231` -> `0.132398` (42.7% lower).
- Line CV: `0.231231` -> `0.244607` (+5.8%).
- Line max density: `2.9625e+18` -> `5.4733e+18` (x1.848).
- Line mean density: `2.3427e+18` -> `4.0727e+18` (x1.738).
- Boundary gamma CV: `0.533656` -> `0.639548` (+19.8%).

## Optimized variables
- `pp=2727.49`, `pp0=0.0997569`.
- Effective coil count: `2`.
- `llcoil=0.889148`, `rrc=5.04697`, `rrce=24.09`, `zzc=3.26605`.

## Interpretation
The selected objective improved mainly by increasing peak and mean electron density. The line CV itself became slightly worse. This is a valid optimum for the chosen scalar objective, but it should be presented as a density-weighted uniformity result, not as a pure uniformity improvement. If the conference story requires uniformity to improve monotonically, rerun with an explicit CV constraint or a penalty for CV worse than the base case. Boundary gamma uniformity also worsened because this run intentionally used the density-weighted objective without a boundary penalty.

## Generated figures
- `line_profile_base_best_linear.png`
- `optimization_scatter_uniformity_density.png`
- `optimization_scatter_uniformity_density_logcolor.png`
- `optimization_coil_count_objective_boxplot.png`
- `optimization_objective_convergence_log.png`
- `optimization_variable_objective_scatter.png`
- `optimization_base_best_metric_bars.png`
- Existing: `trial_profiles_z119_r0_400.png`, `geometry_before_after_arrows.png`, `fields_log_before_after.png`, `qoi_before_after.png`
