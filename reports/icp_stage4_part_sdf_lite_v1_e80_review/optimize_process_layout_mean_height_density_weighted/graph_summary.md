# Mean-Height Density-Weighted Optimization Graph Summary

Objective: minimize `(R-direction relative uniformity) * abs(R-direction mean ne)` at the plasma mean-height row. Conditions `pp/pp0` and coil rectangle layout are optimized.

| model | base score | best score | improvement | base rel | best rel | base mean ne | best mean ne | best pp | best pp0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ffno | 1.0159e+18 | 1.9817e+15 | 99.80% | 0.3544 | 0.3065 | 2.8666e+18 | 6.4646e+15 | 1061.847 | 0.007460 |
| unet | 7.3414e+17 | 6.3110e+16 | 91.40% | 0.8031 | 1.0961 | 9.1412e+17 | -5.7578e+16 | 512.637 | 0.024889 |

Important note: a negative best mean density indicates a nonphysical surrogate escape route for this objective.

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
- `density_weighted_optimization_summary.csv`
- `best_process_conditions.csv`
- `best_layout_parameters.csv`
- `optimization_history_combined.csv`
