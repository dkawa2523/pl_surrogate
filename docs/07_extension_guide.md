# 07 Extension Guide

この guide は contributor が新しい model、target、feature、metric、optimization objective を追加するときの入口だけを示す。

## Principles

- target 定義は `dataset.targets[]` から始める。
- target 順序は `output_layout.vars` を読む。
- target role は `target_role_schema.json` を読む。
- feature 順序は `coord_feature_pack_meta.json` / `channel_map.json` を読む。
- model capability は `src/plasma_surrogate/core/model_specs.py` に置く。
- inference optimization objective は `inference.optimize.objective` に置く。

## Add Or Change A Model

1. `src/plasma_surrogate/core/model_specs.py` に model id と capability を追加する。
2. 必要な最小 model implementation を `src/plasma_surrogate/models/` に追加する。
3. checkpoint build / save / load を接続する。
4. train adapter を追加する。
5. generic inference path で足りない場合だけ inference adapter を追加する。
6. train / infer contract が安定してから benchmark support を追加する。

同じ model-specific policy を train / infer / benchmark に並列で増やさない。

## Add Or Change Targets

1. `dataset.targets[]` に `id`, `source_key`, `role`, metadata を置く。
2. target ごとの transform が必要なら preprocessing config に置く。
3. preprocessing 後の `output_layout.vars` を確認する。
4. physics を使う場合は `physics.symbols` / `inference.ood.*.symbols` を明示する。
5. benchmark metric は active target set から生成する。

## Add Or Change Features

1. preprocessing で feature pack を作る。
2. `coord_feature_pack_meta.json` と `channel_map.json` に順序を残す。
3. inference 側で同じ metadata を読む。
4. required feature が欠けたら fail-fast とする。

feature list を train / infer / benchmark で別々に書かない。

## Add Or Change Metrics

Metrics と compare header は active target から作る。

- `test_rmse_<var>`
- `test_r2_<var>`
- `test_rmse_<var>_plasma`
- `test_r2_<var>_plasma`

Benchmark selection は `surrogate_quality_score` を既定にし、R2 / RMSE は補助指標として扱う。

## Add Or Change Optimization Objective

1. objective config は `inference.optimize.objective` に置く。
2. product mode は `weighted_sum` とする。
3. objective term は QoI / scalar diagnostics から読む。
4. constraints は feasibility と `search_value` に反映する。
5. 新しい objective mode は、現在の scalar objective contract が足りなくなった場合だけ追加する。

能動学習、多様性最適化、Pareto front は未実装の extension point として扱う。

## Tests

通常 CI は小さな contract test を優先する。

- dataset target parsing
- preprocessing artifact roundtrip
- feature channel order
- model spec capability
- train dispatch contract
- inference symbol mapping
- surrogate quality score
- optimization objective smoke

大きな benchmark sweep、生成 config catalog、report reproduction は別 lane で扱う。
