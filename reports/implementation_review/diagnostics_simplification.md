# Diagnostics Simplification Review

## Scope

Reviewed benchmark/eval diagnostic columns introduced for group metrics,
region metrics, positive-target diagnostics, and physics-style diagnostics.
The goal is to keep default leaderboard output compact while preserving opt-in
diagnostic detail for investigation.

## Decision

Default benchmark rows keep only:

- `surrogate_quality_score`
- target-wise `test_rmse_<var>` / `test_r2_<var>`
- plasma target-wise `test_rmse_<var>_plasma` / `test_r2_<var>_plasma`
- group-wise `test_rmse_group_<group>` / `test_r2_group_<group>`
- plasma group-wise `test_rmse_group_<group>_plasma` / `test_r2_group_<group>_plasma`

Extended diagnostics are emitted as top-level dynamic columns only when
`eval.diagnostics.enabled: true`:

- `test_rmse_<var>_boundary_band`
- `test_rmse_<var>_deep_plasma`
- `test_rmse_<var>_outside`
- `test_rmse_group_<group>_boundary_band`
- `test_rmse_group_<group>_deep_plasma`
- `test_rmse_group_<group>_outside`
- `positive_violation_rate_<var>`
- `negative_min_<var>`
- `positive_violation_rate_group_<group>`

Detailed ratios, finite-count checks, older compatibility names, and quality
score components remain under `_diagnostics`, which is written to
`diagnostics/diagnostics.csv` by the benchmark metric table writer.

## Audit Notes

- Group-wise metrics are simple finite means of existing target-wise metrics.
  They are useful as compact summaries for density/temperature/electrostatic
  families and are kept in the default path.
- Region-wise metrics are useful for failure analysis but can multiply columns
  quickly across targets and groups. They now require the single opt-in switch.
- Region masks are still built only when explicit region inputs are available:
  `distance_signed`, or `mask_plasma` plus `distance_any`. No new fallback
  inference was added.
- Positive diagnostics are computed only for
  `target_role_schema.positive_targets` or targets marked `positive: true`.
  They are not calculated for every target.
- Physics residual maps and detailed spatial distribution artifacts remain in
  the existing explicit diagnostics writer path.
- No unused diagnostic helper was removed in this pass; the scoped helpers are
  still used by `core_metrics.py` or benchmark diagnostic writers.

## Follow-Up Candidates

- Consider renaming `eval.diagnostics.enabled` to a more explicit
  `eval.diagnostics.columns` only if future configs need separate control of
  CSV columns versus file artifacts.
- Keep `surrogate_quality_score` internals stable until benchmark comparisons
  justify changing selection behavior.
