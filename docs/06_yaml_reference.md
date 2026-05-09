# 06 YAML Reference

本章は、mainline 実用で触る YAML キーをカテゴリ単位で整理した辞書です。  
すべての内部キーを列挙するのではなく、第三者運用で意味を持つキーに絞ります。

各キーには以下を意識して説明します。

- 何に使うか
- 必須か任意か
- mainline 既定の考え方
- downstream でどこに効くか

## 1. `dataset`

```yaml
dataset:
  type: csv_npz
  root: data/my_dataset
  index_csv: index.csv
  cond_columns: [p0, td, gamma]
  axis_column: axis
  fields_npz_column: fields_npz
  case_id_column: case_id
  base_case_id_column: base_case_id
  split_group_column: split_group
  geometry_root: geometry
  targets:
    - id: density
      source_key: density_raw
      value_transform: identity
      units: "m^-3"
      dtype: float32
```

### 何に使うか

- source データを logical target と条件変数へ対応付ける
- preprocess 以前の truth source を決める

### 必須

- `type`
- `root`
- `cond_columns`
- `targets`

### mainline 既定

- `type=csv_npz`
- target 名は固定しない

### downstream

- `dataset_io`
- split
- preprocess schema
- model dispatch
- metrics / benchmark / compare

### mainline でよく触るキー

- `targets`
- `cond_columns`
- `geometry_root`

### mainline で禁止または非推奨

- `output_vars`
- `output_key_map`
- `output_value_transform`

## 2. `preprocessing`

```yaml
preprocessing:
  scalers:
    enforce_target_transforms: true
    y_fit_policy: plasma_only
    target_transforms:
      density:
        value_transform: log10
        scaler: zscore
        fit_scope: plasma_only
        clip:
          mode: quantile
          q_low: 0.01
          q_high: 0.99
```

### 何に使うか

- logical target を学習しやすい表現へ変換する
- split / scaler / schema / feature artifact を生成する

### `target_transforms.<var>`

#### 用途

- 各 target の個別前処理を定義する

#### 主なキー

- `value_transform`: `identity | log10`
- `scaler`: `none | zscore | minmax`
- `fit_scope`: `all | plasma_only`
- `clip.mode`: `none | quantile`

#### downstream

- train
- infer
- evaluate

### mainline 既定

- target ごとに必ず定義する
- transform 順序は `value_transform -> clip -> scaler`

### mainline で禁止

- `target_transform_policy`

### よく触るキー

- `scalers.target_transforms`
- `coord_features.channels`
- `coord_features.distance_transform_stats`

## 3. `split`

```yaml
split:
  seed: 7
  ratios: [0.7, 0.15, 0.15]
```

### 用途

- train / val / test の分割を固定する

### 必須

- `ratios`

### downstream

- preprocess
- benchmark protocol

## 4. `train`

`train` はモデル共通の loss と、モデル別設定を持ちます。

### 共通 loss

```yaml
train:
  loss:
    supervised:
      type: huber
      delta: 1.0
      mask: plasma_only
```

#### 用途

- teacher supervision の形を決める

#### よく触るキー

- `type`
- `delta`
- `mask`
- `normalization`

### モデル共通で意味のあるキー

- `target_family`
- `target_vars`
- `selection.mode`
- `selection.weights`
- `optimizer`

### `train.global_mlp`

```yaml
train:
  global_mlp:
    epochs: 80
    lr: 8.0e-4
    batch_size_cases: 6
```

#### 用途

- 条件ベクトルから field を直接出す MLP

#### よく触るキー

- `epochs`
- `lr`
- `batch_size_cases`
- `model_cfg.hidden`

### `train.unet`

```yaml
train:
  unet:
    epochs: 80
    target_family: allvars
    target_vars: [density, temperature, potential]
    input_features:
      mode: geom_feature_pack
      require_pack: error
      features: [x, y, mask_plasma, distance_signed, distance_any]
    selection:
      mode: best_val_allvars_balance
```

#### strict contract で意味を持つキー

- `target_family`
- `target_vars`
- `input_features.mode`
- `selection.mode`
- `model_cfg.output_heads.mode`

### `train.fno`

```yaml
train:
  fno:
    epochs: 80
    target_family: allvars
    target_vars: [density, temperature, potential]
    input_features:
      mode: geom_feature_pack
    model_cfg:
      n_modes: 12
      spectral_cfg:
        width: 64
        n_layers: 4
```

#### strict contract で意味を持つキー

- `target_family`
- `target_vars`
- `input_features.mode`
- `selection.mode`

#### 実運用でよく触るキー

- `n_modes`
- `spectral_cfg.width`
- `spectral_cfg.n_layers`
- `spectral_cfg.dealias_ratio`
- `spectral_cfg.taper_alpha`

### `train.deeponet_plasma`

```yaml
train:
  deeponet_plasma:
    strict_mainline: true
    operator_mode: plain
    target_family: allvars
    target_vars: [density, temperature, potential]
    input_features:
      mode: geom_feature_pack
    model_cfg:
      trunk_input_mode: geom_feature_pack
      branch_mode: cond_only
```

#### strict contract で意味を持つキー

- `strict_mainline`
- `operator_mode`
- `target_family`
- `target_vars`
- `input_features.mode`
- `model_cfg.trunk_input_mode`
- `model_cfg.branch_mode`
- `selection.mode`

## 5. `inference`

```yaml
inference:
  single:
    enabled: true
    cond: {c0: 0.2, c1: 0.5, c2: 0.8}
    geom: {geom_id: default}
    axis: {mode: steady, value: 0.0}
  batch:
    enabled: true
    cases:
      - case_id: c001
        cond: {c0: 0.1, c1: 0.2, c2: 0.3}
        geom: {geom_id: default}
        axis: {mode: steady, value: 0.0}
      - case_id: c002
        cond: {c0: 0.4, c1: 0.5, c2: 0.6}
        geom:
          geom_id: default
          geom_param: {part.p0.tx: 0.05}
    csv:
      path: candidates.csv
      cond_columns: [c0, c1, c2]
      case_id_column: case_id
      geom_id_column: geom_id
  ood:
    physics:
      enabled: false
      symbols:
        density: density
```

### 用途

- 学習済み checkpoint と preprocess artifact を使って推論する。
- `single` は 1 条件の確認、`batch` は候補群の推論と QoI/diagnostic 集計に使う。
- `batch.cases` は YAML 直書き、`batch.csv` は CSV 候補を読むための薄い入口。

### 重要キー

- `single.enabled`
- `single.cond`
- `single.geom`
- `single.axis`
- `batch.enabled`
- `batch.conds`: 既存互換の条件リスト。全 case で同じ `batch.geom` / `batch.axis` を使う。
- `batch.cases`: `case_id`, `cond`, `geom`, `axis` を case ごとに指定する。
- `batch.csv.path`
- `batch.csv.cond_columns`
- `batch.csv.case_id_column`
- `batch.csv.geom_id_column`
- `ood.physics.enabled`
- `ood.physics.symbols`
- `ood.boundary_operator.symbols`

### 出力

- `inference/single/<case_key>/fields_model.npz`
- `inference/single/<case_key>/fields_phys.npz`
- `inference/single/<case_key>/qoi.json`
- `inference/single/<case_key>/diagnostics.json`
- `inference/batch/summary.csv`
- `inference/cases_summary.csv`
- `inference/cases_summary.json`

`cases_summary.csv/json` includes `case_key` so each row can be traced back to
`inference/single/<case_key>/...`. Missing QoI/diagnostic values are written as
empty CSV cells and `null` in JSON, not as `0.0`.

For `inference.optimize`, omit `space` only when preprocessing `cond_stats` has
complete `min`/`max` bounds; those bounds are used instead of a synthetic
`[0, 1]` default.

`cases_summary` は case ごとの `case_id`, `cond`, `geom`, `axis`, QoI, diagnostics をまとめる。初回統合では ground truth 比較や metric registry は持たず、推論結果から得られる既存 QoI/diagnostic の集約に留める。

### geometry

- 通常は `geom: {geom_id: default}` を使う。
- `geom.geom_param` は `runtime.input_mode=table_plus_structure` かつ `runtime.structure.provider_mode=parametric_parts` のときだけ有効。
- `geom_param` の詳細 schema は `inference` 側で重ねず、既存 geometry provider の contract に委ねる。

### mainline 既定

- `single.cond` 未指定時は condition schema の各キーに `0.5` を使う。
- `batch.conds` / `batch.cases` / `batch.csv` が未指定なら、既存互換として簡易な 2 条件 batch を作る。
- `axis` 未指定時は preprocess の axis schema から `mode` を取り、`value: 0.0` を使う。
- physics symbol が未解決のまま OOD physics を有効化しない。

## 6. `physics`

```yaml
physics:
  enabled: true
  terms:
    - name: pinn_residual
      weight: 0.1
```

### 用途

- train 側の物理項を registry 方式で有効化する

### 現在の代表 term

- `poisson`
- `boundary`
- `boundary_operator`
- `rho`
- alias: `pinn_residual`, `pino_operator`

### downstream

- `train/physics_terms.py`
- loss composition

## 7. `benchmark`

```yaml
benchmark:
  output_dir: runs/example
  profile: m7_fno_isolated
  eval:
    target_family_for_score: allvars
    target_vars_for_score: [density, temperature, potential]
    primary_metric: test_r2_plasma_mean_dual
```

### 用途

- benchmark 実行と集計ルールを定義する

### よく触るキー

- `profile`
- `output_dir`
- `eval.target_vars_for_score`
- `eval.primary_metric`
- `eval.aggregate_score`
- `eval.region_bands`
- `eval_protocol`

### downstream

- `benchmark/runner.py`
- leaderboard
- resolved_benchmark
- compare 入力

### 注意

- compare / leaderboard は dynamic target 列前提
- `target_vars_for_score` は active target の部分集合または同一集合

## 8. `compare`

```yaml
compare:
  output_dir: runs/example_compare
  objective_metric: auto_primary
  objective_mode: max
  rows:
    - name: model_a
      target_family: allvars
      leaderboard_csv: runs/model_a/leaderboard.csv
      model_id: fno
```

### 用途

- 複数 leaderboard から比較表を作る

### 生成物

- `selected_models_comparison.csv`

### 重要な理解

- ヘッダは固定ではない
- 各 row の持つ dynamic target 列から最終列集合が決まる

## 9. mainline で避けるべき設定

- target 名のハードコード
- `target_transform_policy`
- fixed compare header を前提にした拡張
- symbol mapping なしの physics
- preprocess artifact を使わない feature 契約
