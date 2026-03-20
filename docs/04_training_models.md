# 04 Training Models

## 1. 学習契約の入口

mainline の学習契約は `src/plasma_surrogate/train/model_dispatch.py` に集約されています。  
trainer はこの validation を通過した構成だけを実行します。

共通前提:

- `target_family=allvars`
- `target_vars == output_layout.vars`
- `selection.mode=best_val_allvars_balance`
- `selection.weights` は明示指定または target 数に応じた均等化

重要なのは、target 名が固定ではないことです。  
`ne / ni / Te / phi` は m7 の例であり、strict 契約そのものではありません。

## 2. モデル比較の見方

| Model | 入力 | geometry の使い方 | 空間依存の扱い | 向くケース |
|---|---|---|---|---|
| `global_mlp` | 条件ベクトル | 直接は使わない | field を直接出力 | 基準モデル、条件依存の大局傾向確認 |
| `unet` | 条件 + grid feature | 強い | 局所畳み込み | 境界形状が重要な固定 grid 問題 |
| `fno` | 条件 + spatial feature | 強い | spectral operator | 広域依存が強い fixed grid 問題 |
| `deeponet_plasma` | 条件 + query/geometry feature | trunk 側で強い | operator 形 | 条件依存な operator として扱いたい問題 |

## 3. `global_mlp`

### 入力形

- 条件ベクトル

### 出力形

- fixed grid 上の target field

### geometry / feature の扱い

- mainline では geometry feature を直接入力しない
- そのため空間形状の複雑さより、条件依存の平均的な傾向把握に向く

### 主な YAML 入口

```yaml
train:
  global_mlp:
    epochs: 80
    lr: 8.0e-4
    batch_size_cases: 6
    model_cfg:
      hidden: [128, 128]
      dropout: 0.0
```

### 典型的な落とし穴

- geometry が効かない前提なのに境界詳細の再現を期待しすぎる
- target 数が増えても hidden 幅を据え置いてボトルネックにする

## 4. `unet`

### 入力形

- 条件ベクトルを grid に broadcast
- `coord_feature_pack` 由来の spatial feature をチャンネル結合

### 出力形

- 同一 grid の target field

### geometry / feature の扱い

- geometry 情報を spatial channel として直接使う
- 局所構造の扱いに強い

### mainline strict contract

- `train.unet.target_family=allvars`
- `train.unet.target_vars == output_layout.vars`
- `train.unet.input_features.mode=geom_feature_pack`
- `train.unet.model_cfg.output_heads.mode=shared`
- `train.unet.selection.mode=best_val_allvars_balance`

### 主な YAML 入口

```yaml
train:
  unet:
    epochs: 80
    optimizer:
      type: adamw
      schedule: cosine
    input_features:
      mode: geom_feature_pack
      require_pack: error
      features: [x, y, mask_plasma, distance_signed, distance_any]
    model_cfg:
      backend: torch
      base_channels: 32
      output_heads:
        mode: shared
```

### 典型的な落とし穴

- `input_features.mode` と preprocess artifact がずれる
- target 数変更時に `target_vars` を更新しても `weights` を放置する
- `split_density_field` のような mainline 外設定を紛れ込ませる

## 5. `fno`

### 入力形

- 条件ベクトル
- `geom_feature_pack` 由来の spatial feature

### 出力形

- fixed grid の operator 出力

### geometry / feature の扱い

- geometry を spectral operator の入力チャネルとして使う
- 局所だけでなく広域依存を扱いやすい

### mainline strict contract

- `train.fno.target_family=allvars`
- `train.fno.target_vars == output_layout.vars`
- `train.fno.input_features.mode=geom_feature_pack`
- 必須チャネルは `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any`
- `selection.mode=best_val_allvars_balance`

### 主な YAML 入口

```yaml
train:
  fno:
    epochs: 80
    optimizer:
      type: adamw
      schedule: cosine
    input_features:
      mode: geom_feature_pack
      require_pack: error
    model_cfg:
      n_modes: 12
      spectral_cfg:
        width: 64
        n_layers: 4
        dealias_ratio: 0.67
        taper_alpha: 4.0
        skip_filter: match_spectral
```

### 典型的な落とし穴

- `legacy_xy` 的な入力を mainline と混同する
- `distance_transform_stats` を使う runtime 設定なのに preprocess を対応させない
- target 追加時に selection や benchmark 側の評価 target を追随させない

## 6. `deeponet_plasma`

### 入力形

- branch: 条件ベクトル
- trunk: query/grid 位置 + geometry/coord feature

### 出力形

- 各 query または grid 点における target field

### geometry / feature の扱い

- trunk 側で geometry / coord feature を使う
- 条件依存 operator として出力分布を作る

### mainline strict contract

`model_dispatch` 上の制約が特に重要です。

- `strict_mainline=true`
- `operator_mode=plain`
- `input_features.mode=geom_feature_pack`
- `model_cfg.trunk_input_mode=geom_feature_pack`
- `model_cfg.branch_mode=cond_only`
- `target_family=allvars`
- `target_vars == output_layout.vars`
- `selection.mode=best_val_allvars_balance`

### 主な YAML 入口

```yaml
train:
  deeponet_plasma:
    strict_mainline: true
    epochs: 160
    optimizer:
      type: adamw
      schedule: cosine
    input_features:
      mode: geom_feature_pack
      require_pack: error
      features: [x, y, mask_plasma, distance_signed, distance_any]
    model_cfg:
      latent_dim: 48
      hidden_dim: 96
      trunk_input_mode: geom_feature_pack
      branch_mode: cond_only
      trunk_fourier_n_freq: 4
      trunk_fourier_mode: symmetric
      output_path:
        mode: dot
```

### 典型的な落とし穴

- `strict_mainline` に対して legacy trunk / branch 設定を混ぜる
- branch/trunk の役割を取り違えて feature 設定を置く
- target 名を変えたのに physics symbol 側を更新しない

## 7. `selection=best_val_allvars_balance`

### 何を見ているか

active target 群をまとめて扱う validation ベースの selection です。  
個別 target の良し悪しを単純に 1 変数だけで選ぶのではなく、target 群全体のバランスを見ます。

### target 非固定でどう動くか

- `target_vars` に含まれる target 全体を対象とする
- target 数が変わっても構成は維持される

### `weights` の解釈

- 明示指定があればその重みを使う
- 未指定なら target 数に応じて均等化する

### 注意点

- target を追加・削除したときに `weights` が古いままだと validation mismatch を起こす

## 8. trainer と `model_dispatch` の責務分離

### `model_dispatch`

- 設定が mainline 契約を満たすかを判定する
- target 順序、feature 契約、許可/禁止設定を固める

### trainer

- optimizer を回す
- loss を計算する
- diagnostics を出力する

この分離により、第三者が設定トラブルを見るときは、まず `model_dispatch` から読むのが最短です。

## 9. 修正入口

- モデル構造を変える: `src/plasma_surrogate/models/*`
- build/save/load 経路: `src/plasma_surrogate/models/mlp/io.py`
- strict contract を変える: `src/plasma_surrogate/train/model_dispatch.py`
- 学習実行を変える: `src/plasma_surrogate/train/trainer.py`, `src/plasma_surrogate/train/torch_trainer.py`
