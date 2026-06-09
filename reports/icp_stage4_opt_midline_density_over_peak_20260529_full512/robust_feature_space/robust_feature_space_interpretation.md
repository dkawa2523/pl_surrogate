# Robust feature-space interpretation

- Finite trials: 512. Top robust basin is the best 52 trials (top 10%).
- Best objective: 0.132398. Best line CV: 0.244607. Best peak-density ratio: 1.848.
- Top-10% effective coil-count distribution: {2: 10, 3: 11, 4: 11, 5: 9, 6: 11}.
- These figures use process variables and derived coil-series descriptors as the feature-space coordinates. The surrogate still receives structure feature maps, not raw geometry parameters.
- Many high-ranking designs push `pp` beyond the train split range warning; treat them as extrapolative COMSOL candidates unless a train-bounds-only rerun confirms them.

## Top-10% robust ranges

| feature | q10 | median | q90 |
|---|---:|---:|---:|
| `pp` | 1922.2 | 2509 | 2941.4 |
| `pp0` | 0.088598 | 0.095536 | 0.099565 |
| `series.nncoil_effective` | 2 | 4 | 6 |
| `series.llcoil` | 0.61627 | 0.85972 | 1.117 |
| `series.rrc` | 2.9521 | 5.1126 | 7.5666 |
| `series.rrce` | 20.799 | 22.883 | 25.342 |
| `series.zzc` | 0.41936 | 2.1672 | 3.9184 |
| `series.pitch_effective` | 2.8964 | 4.3157 | 9.3566 |
| `series.min_gap_effective` | 2.195 | 3.5176 | 8.4271 |
| `series.span_r` | 13.401 | 17.803 | 20.81 |
| `series.fill_ratio` | 0.098548 | 0.19383 | 0.31962 |
| `series.total_coil_area_proxy` | 1.1842 | 2.4407 | 5.9054 |

## Figures
- `feature_space_pca_objective.png`: global feature-space map colored by objective.
- `feature_space_pca_density_uniformity.png`: separates peak-density gain and line CV.
- `robust_feature_range_boxplots.png`: all trials vs top-10%/top-5% ranges.
- `top20_coil_layout_overlay.png`: top coil layouts, best in red.
- `structure_descriptor_objective_panels.png`: interpretable layout descriptors vs objective.
- `top20_objective_decomposition.png`: objective/CV/density decomposition for top candidates.