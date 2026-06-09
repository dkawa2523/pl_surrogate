# Sampler comparison: parameter optimization vs structure-feature archive

## What is compared
- Parameter two-stage: random global search followed by local sampling around low-objective parameter candidates.
- Feature archive: generate a large candidate pool, convert candidates to derived coil-structure descriptors, deduplicate/select by descriptor-space coverage, then evaluate the selected unique structures.
- In both cases, the surrogate receives generated structure maps/features, not raw geometry vectors. The difference is how candidates are selected before inference.

## Key numbers
| sampler | n_trials | best_value | best_cv | best_density_ratio | best_nncoil | top10_median_objective | top10_median_cv | top10_median_density_ratio | all_median_nn_dist | top10_median_nn_dist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature archive | 512 | 0.11742 | 0.26757 | 2.2787 | 6 | 0.16312 | 0.26086 | 1.6097 | 1.9196 | 2.2128 |
| parameter two-stage | 512 | 0.1324 | 0.24461 | 1.8475 | 2 | 0.15938 | 0.25746 | 1.6559 | 1.2986 | 0.87461 |

## Interpretation
- Feature archive reached a lower best scalar objective and higher density-ratio candidates than two-stage.
- Parameter two-stage is more exploitative/local; feature archive preserves broader structural coverage and gives more diverse COMSOL candidate families.
- The density-weighted objective still favors high-density designs even when CV worsens, so the Pareto plot is the most important design-selection figure.
- For third-party explanation, use the feature-space coverage, nearest-neighbor distance, and coil-count balance plots to show why structure-feature-aware candidate selection is useful.

## Figures
- `sampler_comparison_objective_history_log.png`
- `sampler_comparison_cv_density_tradeoff.png`
- `sampler_comparison_feature_space_coverage.png`
- `sampler_comparison_nearest_descriptor_distance.png`
- `sampler_comparison_coil_count_balance.png`
- `sampler_comparison_summary_bars.png`
- `sampler_comparison_top20_layout_overlay.png`
- `sampler_comparison_nearest_descriptor_distance_box.png`
- `sampler_comparison_feature_archive_benefit_ratios.png`