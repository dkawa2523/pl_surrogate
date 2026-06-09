# Wide-Range Balanced Optimization Graph Summary

Objective: minimize `CV_R(ne) / sqrt(density_gain) + negative-density penalty + low-density penalty` at the plasma mean-height row.
Search space: `pp/pp0` dataset bounds, coil center +/-0.90, coil width/height 0.50-1.50x, 240 random trials per model.

| model | base score | best score | improvement | base CV | best CV | base mean ne | best mean ne | density gain | best pp | best pp0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ffno | 0.3544 | 0.3069 | 13.41% | 0.3544 | 0.3212 | 2.8666e+18 | 3.1403e+18 | 1.095 | 2827.941 | 0.095009 |
| unet | 0.8031 | 0.7022 | 12.57% | 0.8031 | 0.7734 | 9.1412e+17 | 1.1089e+18 | 1.213 | 2676.517 | 0.096224 |

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
- `wide_optimization_summary.csv`
- `balanced_optimization_summary.csv`
- `best_process_conditions.csv`
- `best_layout_parameters.csv`
- `optimization_history_combined.csv`
