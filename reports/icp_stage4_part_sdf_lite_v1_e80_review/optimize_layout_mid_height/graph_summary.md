# Optimization Graph Summary

Objective: minimize `ne` uniformity on the plasma mid-height line. Process conditions `pp/pp0` are fixed.

| model | base | best | improvement | trials | invalid |
|---|---:|---:|---:|---:|---:|
| ffno | 0.3396 | 0.3204 | 5.65% | 80 | 0 |
| unet | 0.7155 | 0.6994 | 2.26% | 80 | 0 |

Key plots:
- `plots/optimization_history_combined.png`
- `plots/optimization_improvement_bar.png`
- `plots/spatial_distribution_base_best_ne_combined.png`
- `plots/spatial_distribution_base_best_ni_combined.png`
- `plots/mid_height_ne_profile_combined.png`
- `plots/mid_height_ni_profile_combined.png`
- `plots/ffno_best_layout_delta_heatmap.png`
- `plots/unet_best_layout_delta_heatmap.png`

CSV outputs:
- `optimization_history_combined.csv`
- `layout_mid_height_optimization_summary.csv`
- `best_layout_parameters.csv`
