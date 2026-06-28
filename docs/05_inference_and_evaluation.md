# 05 Inference And Evaluation

Inference connects a checkpoint with preprocessing artifacts. Evaluation reports
both metric values and whether those values are valid for selection.

## Inference Contract

`InferenceEngine` reads:

- target order and target roles
- target transforms
- feature order and channel map
- runtime schema hashes
- model capability metadata

Checkpoint metadata and request metadata must match for required runtime keys.

## Physics Symbols

Physics, OOD diagnostics, and boundary-operator diagnostics resolve targets from
explicit symbols first, then unique target role/family metadata. Target names are
examples, not aliases baked into the product contract.

```yaml
physics:
  symbols:
    density: electron_density
    temperature: electron_temperature
    potential: plasma_potential
  terms:
    poisson:
      weight: 0.1
```

## Derived Fields

Learned targets are checkpoint outputs listed in `output_layout.vars`. Derived
fields are inference-time products computed from physical-space learned targets.
They are not training targets and are not fed into loss.

By default, inference writes `E_mag` only when a unique potential can be
resolved from `physics.symbols` or target-role metadata. Optional
`inference.derived_fields` entries can request `negative_gradient`,
`vector_magnitude`, or `electric_field_magnitude`. Missing sources are skipped by
default; set `derived_fields_strict: true` to fail fast.

## Metrics

Evaluation produces dynamic target columns:

- `test_rmse_<var>`
- `test_r2_<var>`
- `test_rmse_<var>_plasma`
- `test_r2_<var>_plasma`

Masked metrics return `NaN` when the active mask is empty or active values are
non-finite. This prevents invalid regions from looking like perfect scores.

## Benchmark Tables

Benchmark writes comparison and diagnostics separately:

- `leaderboard.csv`: model id, target RMSE/R2, `surrogate_quality_score`, and
  primary metric reliability.
- `core_metrics.csv`: the same core comparison columns without diagnostic
  payloads.
- `diagnostics/diagnostics.csv`: quality components, finite-count checks,
  boundary/deep-region contrast, continuity ratios, and optional distribution
  summaries.

`target_metrics_valid` stays in the core row because it gates primary metric
reliability. Detailed `quality_components` and `validity_flags` live in the
diagnostics table.

`surrogate_quality_score` remains the default lower-better selection metric. Its
sign component uses `target_role_schema.positive_targets`; without positive
metadata, that component contributes zero.

## Optimization Objective

The product objective mode is `weighted_sum`. Terms read scalar QoI first and
then scalar diagnostics. Constraints affect feasibility and `search_value`.
Benchmark summaries expose `status`, `skip_reason`, `objective_value`,
`search_value`, and constraint violation information.
