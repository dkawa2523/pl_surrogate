# Optimization Graphs

Generated from existing ICP_stage4 part_sdf_lite_v1 optimization reports. No new optimization run was started.

## Files
- `optimization_summary.csv`: parsed base/best QoI summary.
- `best_uniformity_by_scenario.png`: raw comparison including outlier objectives.
- `uniformity_improvement_pct.png`: raw improvement comparison including outlier objectives.
- `best_so_far_curves.png`: raw best-so-far curves.
- `best_uniformity_by_scenario_filtered.png`: practical comparison with density-weighted outliers excluded.
- `uniformity_improvement_pct_filtered.png`: filtered improvement comparison.
- `ffno_selected_best_so_far_curves.png`: selected FFNO objective curves.
- `ffno_process_layout_mid_height_geometry_before_after.png`: lowest mid-height uniformity geometry before/after.
- `ffno_process_layout_mid_height_fields_before_after.png`: lowest mid-height uniformity field before/after.
- `ffno_process_layout_mean_height_balanced_wide_geometry_before_after.png`: density-preserving balanced-wide geometry before/after.
- `ffno_process_layout_mean_height_balanced_wide_fields_before_after.png`: density-preserving balanced-wide field before/after.

- `geometry_delta_summary.csv`: numeric center/endpoint/size deltas for selected FFNO layouts.
- `geometry_delta_summary.png`: delta-focused view of center and size changes.

## Reading Notes
- The mid-height FFNO process+layout case reaches the lowest uniformity value, but its predicted density field collapses strongly; do not treat it as a final physical design without density/penalty constraints.
- The balanced-wide FFNO case has a smaller uniformity improvement but preserves/increases mean density, making it the safer candidate for follow-up.
- Existing density-weighted objective runs are plotted only in raw charts because their objective scale is an outlier and not directly comparable.
