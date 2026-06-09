# Feature-archive optimization evaluation

## Sampler
- Generated 4096 valid coil/process candidates, converted each candidate to a derived structure descriptor, then selected 512 unique feature signatures by farthest-point coverage.
- The surrogate still receives generated structure maps/features; raw geometry parameters are not used as model inputs.
- Unique selected feature signatures: 512 / 512.

## Best objective result
- Objective: `0.231231` -> `0.117422` (49.2% lower than base).
- Previous two-stage best objective: `0.1323982968063745`.
- Line CV: `0.231231` -> `0.267567` (+15.7%).
- Line max density: `2.9625e+18` -> `6.7506e+18` (x2.279).
- Line mean density: `2.3427e+18` -> `4.7190e+18` (x2.014).

## Best variables
- `pp=2777.06`, `pp0=0.0994045`.
- effective coil count `6`, `llcoil=0.9322`, `rrc=3.45572`, `rrce=20.6735`, `zzc=0.830844`.

## Interpretation
- Feature-archive sampling found a lower scalar objective than the previous two-stage run and preserved structural diversity across coil counts.
- The best scalar objective is again density-gain dominated: line CV worsened while peak/mean density rose strongly.
- For design selection, use the Pareto CSV as well as the scalar best. It contains candidates with better CV/density trade-offs that may be more physically attractive for COMSOL reruns.

## Figures
- `feature_archive_objective_history_log.png`
- `feature_archive_line_profile_base_best_linear.png`
- `feature_archive_pareto_cv_density.png`
- `feature_archive_pca_objective.png`
- `feature_archive_interpretable_planes.png`
- `feature_archive_coil_count_success.png`
- `feature_archive_top20_coil_layout_overlay.png`