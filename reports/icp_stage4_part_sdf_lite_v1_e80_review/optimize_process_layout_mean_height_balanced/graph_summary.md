# Balanced Mean-Height Optimization Graph Summary

Objective: minimize `CV_R(ne) / sqrt(density_gain) + negative-density penalty + low-density penalty` at the plasma mean-height row.
The density reference is the base/reference mean-height R-direction mean `ne` for each model.

| model | base score | trial best score | recommended | base CV | trial CV | base mean ne | trial mean ne | density gain | best pp | best pp0 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ffno | 0.3544 | 0.4054 | base | 0.3544 | 0.3893 | 2.8666e+18 | 2.6426e+18 | 0.9219 | 1861.030 | 0.097240 |
| unet | 0.8031 | 1.002 | base | 0.8031 | 0.9482 | 9.1412e+17 | 8.1937e+17 | 0.8963 | 1861.030 | 0.097240 |

Important note: in this 80-trial random search, the best sampled process/layout design did not beat the base design for either model under the balanced objective. This is a useful guardrail result: the previous density-collapse route is suppressed, but the current search needs either more directed/local search or a better surrogate before accepting a changed structure.

Key plots:
- `plots/optimization_history_combined.png`
- `plots/objective_components_base_best.png`
- `plots/best_process_conditions.png`
- `plots/spatial_distribution_base_best_ne_combined.png`
- `plots/mean_height_ne_profile_combined.png`
- `plots/best_comsol_structure_combined.png`
- `plots/ffno_best_comsol_structure.png`
- `plots/unet_best_comsol_structure.png`

CSV outputs:
- `balanced_optimization_summary.csv`
- `best_process_conditions.csv`
- `best_layout_parameters.csv`
- `optimization_history_combined.csv`
