# 20. F-FNO 実装仕様

## 1. 目的

既存 `fno` の外部契約を維持したまま、**少量データでも深い spectral stack を安定に使える factorized FNO** を追加する。

追加 model_id:
- `ffno`

## 2. mainline-candidate として守る条件

`ffno` は `fno` と同じ strict contract を守る。

必須:
- `train.ffno.target_family=allvars`
- `train.ffno.target_vars == output_layout.vars`
- `train.ffno.input_features.mode=geom_feature_pack`
- 既定 feature は `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any`
- `train.ffno.selection.mode=best_val_allvars_balance`

## 3. 変更対象ファイル

### 新規作成
- `src/plasma_surrogate/models/fno/factorized_fno.py`

### 変更
- `src/plasma_surrogate/models/mlp/io.py`
- `src/plasma_surrogate/train/model_dispatch.py`
- unit / integration tests
- benchmark fixture

## 4. モデル設計

## 4.1 外部 wrapper API

`FFNOBaseline` は `FNOBaseline` と同じ API を持つ。

必須メソッド:
- `set_static_spatial_features`
- `_feature_map`
- `forward`
- `predict_fields`
- `backward_raw`
- `state_dict_numpy`
- `load_state_dict_numpy`

最も安全なのは、`simple_fno.py` の wrapper をほぼ踏襲し、内部ネットワークだけ差し替えること。

## 4.2 v1 の factorization 方針

最初の PR では、F-FNO の完全一般化より **repo に載せやすい最小構成**を優先する。

推奨実装:
- 入力: `cond map + static spatial features`
- `1x1 conv` で width 次元へ project
- 各 spectral block で
  - 高さ方向の factorized spectral operator
  - 幅方向の factorized spectral operator
  - 1x1 skip
  - 加算して GELU
- 最後に post head

### block の概念
```python
h = in_proj(x)
for block in blocks:
    h = block(h)
out = post(h)
```

### block の概念式
```python
spec_h = spectral_h(h)
spec_w = spectral_w(h)
skip = pointwise_skip(h)
h = gelu(spec_h + spec_w + skip)
```

## 4.3 config 方針

`fno` の既存キーを最大限再利用し、追加キーだけ増やす。

### 使う既存キー
- `n_modes`
- `spectral_cfg.width`
- `spectral_cfg.n_layers`
- `spectral_cfg.dealias_ratio`
- `spectral_cfg.taper_alpha`
- `spectral_cfg.skip_filter`

### 追加キー
```yaml
spectral_cfg:
  factorized_cfg:
    enabled: true
    mode: separable_1d
    share_weights: false
```

v1 では `mode=separable_1d` だけでよい。

## 5. `models/mlp/io.py` 実装

### `build_model_from_name` に追加

```python
if name == "ffno":
    return FFNOBaseline(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        with_rho_eff_head=with_rho,
        n_modes=int(cfg.get("fno_n_modes", cfg.get("n_modes", 2))),
        head_mlp=dict(cfg.get("head_mlp", {})),
        input_feature_channels=list(unet_feature_channels or ["x", "y"]),
        spectral_cfg=dict(cfg.get("spectral_cfg", {})),
        seed=int(seed),
    )
```

### checkpoint meta
`fno` と同型の meta を使う。

例:
```json
{
  "model_type": "ffno",
  "input_dim": 12,
  "grid_shape": [64, 64],
  "out_channels": 4,
  "output_keys": ["ne", "ni", "Te", "phi"],
  "input_feature_channels": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
  "with_rho_eff_head": false,
  "n_modes": 12,
  "spectral_cfg": {
    "width": 64,
    "n_layers": 4,
    "dealias_ratio": 0.67,
    "taper_alpha": 4.0,
    "skip_filter": "match_spectral",
    "factorized_cfg": {"enabled": true, "mode": "separable_1d"}
  }
}
```

`load_mlp_checkpoint` に `model_type == "ffno"` を追加する。

## 6. `model_dispatch.py` 実装

`ffno` は `fno` と同じ **unet-like torch branch** に組み込む。

### 実装ルール
- `model_name in {"unet", "fno", "ffno", ...}` の group に入れる
- `input_features.mode` は `legacy_xy` を許さず、mainline-candidate では `geom_feature_pack` 前提にする
- `_validate_unet_like_mainline_contract` を `ffno` にも適用する
- `require_pack` / distance transform / coord scaling は `fno` と同じ helper を使う

### strict validation
`ffno` は `fno` と同一の validation でよい。必要ならエラーメッセージだけ `ffno` 表記にする。

## 7. YAML 断片仕様

`configs/experimental/same_fidelity_low_data/ffno_template.yaml` を参照。

最低限:
```yaml
model:
  name: ffno

train:
  ffno:
    target_family: allvars
    target_vars: [<set-to-output_layout.vars>]
    epochs: 80
    optimizer:
      type: adamw
      schedule: cosine
    input_features:
      mode: geom_feature_pack
      require_pack: error
      features: [x, y, mask_plasma, distance_signed, distance_any]
    model_cfg:
      n_modes: 12
      spectral_cfg:
        width: 64
        n_layers: 4
        dealias_ratio: 0.67
        taper_alpha: 4.0
        skip_filter: match_spectral
        factorized_cfg:
          enabled: true
          mode: separable_1d
    selection:
      mode: best_val_allvars_balance
```

## 8. テスト追加

### unit
- `tests/unit/models/test_ffno_model.py`
  - build
  - feature map shape
  - forward shape
  - checkpoint roundtrip

- `tests/unit/train/test_model_dispatch_ffno.py`
  - strict contract
  - `geom_feature_pack` channel validation

### integration
- `tests/integration/test_train_ffno_smoke.py`
- `tests/integration/test_infer_ffno_smoke.py`
- 必要なら `tests/integration/test_cli_ffno_smoke.py`

## 9. benchmark fixture 名

mainline-candidate として:
- `tests/fixtures/benchmark_periodic_real_m7_ffno_isolated_mainline.yaml`

既存 `m7_fno_isolated` と比較できるよう、可能なら fixture 構造を揃える。

## 10. 受け入れ条件

- 既存 `fno` が壊れない
- `ffno` の build/load/train/infer smoke が通る
- `target_vars == output_layout.vars`
- dynamic benchmark / compare を壊さない
- feature pack の順序が train / infer で一致する
