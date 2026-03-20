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
  axis:
    mode: steady
    value: 0.0
  ood:
    physics:
      enabled: false
      symbols:
        density: density
```

### 用途

- 推論時の軸設定
- OOD / physics / boundary operator の解釈

### 重要キー

- `axis.mode`
- `axis.value`
- `ood.physics.enabled`
- `ood.physics.symbols`
- `ood.boundary_operator.symbols`

### mainline 既定

- symbol 未解決のまま physics を有効化しない

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
