# Process + Structure Optimization Graph Summary

Objective: minimize `ne` uniformity on the plasma mid-height line. Conditions `pp/pp0` and coil rectangle layout are optimized.

| model | base | best | improvement | best pp | best pp0 | trials | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| ffno | 0.3396 | 0.2538 | 25.26% | 1061.847 | 0.007460 | 80 | 0 |
| unet | 0.7155 | 0.3937 | 44.98% | 1323.162 | 0.004081 | 80 | 0 |

Key plots:
- `plots/optimization_history_combined.png`
- `plots/optimization_improvement_bar.png`
- `plots/best_process_conditions.png`
- `plots/spatial_distribution_base_best_ne_combined.png`
- `plots/spatial_distribution_base_best_ni_combined.png`
- `plots/mid_height_ne_profile_combined.png`
- `plots/best_comsol_structure_combined.png`
- `plots/ffno_best_comsol_structure.png`
- `plots/unet_best_comsol_structure.png`

CSV outputs:
- `process_layout_optimization_summary.csv`
- `best_process_conditions.csv`
- `best_layout_parameters.csv`
- `optimization_history_combined.csv`
