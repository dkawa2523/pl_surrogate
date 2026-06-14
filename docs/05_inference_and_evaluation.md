# 05 Inference And Evaluation

inference / evaluation は checkpoint と preprocessing artifact を接続し、field、QoI、metrics、benchmark summary を作る。

## Inference Contract

`InferenceEngine` は次を正本として読む。

- target order: `preprocessing/schema/output_layout.json`
- target role metadata: `preprocessing/schema/target_role_schema.json`
- target transforms: `preprocessing/scalers/y_scalers.json`
- feature order: `preprocessing/features/coord_feature_pack_meta.json`
- channel order: `preprocessing/schema/channel_map.json`
- runtime schema hash: `preprocessing/validation/runtime_schema_hashes.json`
- model capability: `src/plasma_surrogate/core/model_specs.py`

checkpoint metadata と runtime request metadata の required keys が一致しない場合は fail-fast とする。

## Physics Symbol Mapping

physics / OOD / boundary operator は target 名 alias に依存しない。明示 `symbols` を最優先し、無い場合だけ一意の role / field family から解決する。解決不能または曖昧な場合は fail-fast とする。

```yaml
inference:
  ood:
    physics:
      enabled: true
      symbols:
        density: electron_density
        temperature: electron_temperature
        potential: plasma_potential
```

## Metrics

evaluate / benchmark は active target に応じた dynamic metric column を作る。

- `test_rmse_<var>`
- `test_r2_<var>`
- `test_rmse_<var>_plasma`
- `test_r2_<var>_plasma`

固定 target header は product contract にしない。

## Benchmark Selection

benchmark selection の default は lower-better の `surrogate_quality_score` である。R2 / RMSE は補助指標として残すが、primary selection にはしない。

`surrogate_quality_score` は次の component を集約する。

- target ごとの normalized RMSE
- boundary / deep region の誤差バランス
- continuity diagnostic
- physics residual diagnostic
- positive target の sign penalty

role schema が無い場合、sign penalty は 0 contribution とする。

## Optimization Objective

`inference.optimize.objective` の product contract は `weighted_sum` のみである。各 term は `InferenceResult.qoi` を先に参照し、無ければ `InferenceResult.diagnostics` を参照する。

```yaml
inference:
  optimize:
    enabled: true
    backend: optuna
    objective:
      mode: weighted_sum
      terms:
        - key: uniformity
          direction: min
          weight: 1.0
        - key: boundary_gamma_uniformity
          direction: min
          weight: 0.3
        - key: poisson_residual_norm
          direction: min
          weight: 0.2
          transform: log1p_abs
          scale: 1.0
    constraints:
      - key: poisson_residual_norm
        upper: 0.05
```

Term fields:

- `key`: QoI または scalar diagnostic 名。
- `direction`: `min` または `max`。`max` は lower-better objective へ負寄与で変換する。
- `weight`: weighted sum の係数。
- `scale`: 正の有限値。既定は `1.0`。
- `transform`: `identity` または `log1p_abs`。

missing / non-finite term は config error として fail-fast とする。Best selection は feasible trial を優先し、feasible trial が無い場合だけ全 trial の最小 `objective_value` を使う。

Trial record は `objective_value`, `search_value`, `feasible`, `violated_constraints`, `constraint_violation_total`, `qoi_*`, `diagnostic_*`, `objective_term_*` を持つ。

## Planned Extension Points

能動学習、多様性最適化、Pareto front は未実装である。将来は `objective.mode` を追加して拡張し、現行の `weighted_sum` contract を肥大化させない。
