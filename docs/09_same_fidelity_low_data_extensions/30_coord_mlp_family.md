# 30. CoordMLP Family 実装仕様

## 1. 目的

既存 `global_mlp` の弱点である **geometry を直接見ない** 点を補うため、`coord_feature_pack` を使う座標デコーダ型 MLP を追加する。

追加 model_id:
- `coord_mlp_fourier`
- `coord_mlp_siren`

これらは最初は **experimental** とする。

## 2. 実装優先順

### Phase 2a
- `coord_mlp_fourier`

### Phase 2b
- `coord_mlp_siren`

SIREN は Fourier 版の API と builder / dispatch が安定してから追加する。

## 3. 変更対象ファイル

### 新規作成
- `src/plasma_surrogate/models/mlp/coord_mlp_torch.py`

### 変更
- `src/plasma_surrogate/models/checkpoint.py`
- `src/plasma_surrogate/train/model_dispatch.py`
- unit / integration tests
- 必要なら `docs/04_training_models.md` など仕様追記

## 4. 実装方針

## 4.1 重要方針

- **NumPy の `global_mlp` 路線には乗せない**
- `unet` / `fno` と同じ **torch-style wrapper** にする
- `coord_feature_pack` の static spatial feature を使う
- v1 は **full-grid decode** のみでよい
- point sampling / chunked decode は後回し

## 4.2 学習 branch の置き場

`coord_mlp_fourier` / `coord_mlp_siren` は、`global_mlp` の NumPy branch ではなく、
**unet-like torch branch** に入れる。

理由:
- feature pack 解決ロジックを再利用できる
- `set_static_spatial_features` を使える
- `backward_raw` / checkpoint 実装を `unet` / `fno` と揃えやすい

## 5. モデル設計

## 5.1 共通 wrapper API

```python
class CoordMLPTorch:
    def __init__(
        self,
        *,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int,
        output_keys: list[str] | None,
        input_feature_channels: list[str],
        model_cfg: dict[str, Any],
        seed: int,
    ) -> None: ...
```

必須メソッド:
- `set_static_spatial_features`
- `_resolve_spatial_features`
- `_build_decoder_input`
- `forward`
- `predict_fields`
- `backward_raw`
- `state_dict_numpy`
- `load_state_dict_numpy`

## 5.2 共通 forward 構造

```text
cond vector -> cond encoder -> z_cond
coord_feature_pack[H,W,C] -> point embedding -> z_point
concat(z_cond, z_point) -> point decoder -> field values
reshape -> [B, C_out, H, W]
```

### cond encoder
推奨:
- MLP
- hidden 2〜3 層
- GELU or Tanh
- 出力は `latent_dim`

### point decoder
推奨:
- hidden 3 層程度
- 各点で multi-target を shared で出す

## 5.3 Fourier 版

### 入力 embedding
`coord_feature_pack` の各 scalar channel を Fourier embedding する。

v1 は deterministic でよい。

推奨:
```text
for each scalar s:
  [s,
   sin(2^0 * pi * s / scale), cos(2^0 * pi * s / scale),
   sin(2^1 * pi * s / scale), cos(2^1 * pi * s / scale),
   ...]
```

config 例:
```yaml
embedding:
  type: fourier
  n_frequencies: 8
  include_raw: true
  frequency_scale: 10.0
```

## 5.4 SIREN 版

SIREN は decoder の活性関数と初期化だけ変える。

config 例:
```yaml
embedding:
  type: none
siren:
  enabled: true
  w0_initial: 30.0
  w0_hidden: 1.0
```

### 実装上の注意
- first layer は `w0_initial`
- hidden layer は `w0_hidden`
- 初期化は SIREN 用の初期化を実装する
- `coord_mlp_fourier` と同じ wrapper API を維持する

## 6. `models/checkpoint.py` 実装

### `build_model_from_name`

```python
if name in {"coord_mlp_fourier", "coord_mlp_siren"}:
    cfg_local = dict(cfg)
    if name == "coord_mlp_siren":
        siren_cfg = dict(cfg_local.get("siren", {}))
        siren_cfg["enabled"] = True
        cfg_local["siren"] = siren_cfg
    return CoordMLPTorch(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        input_feature_channels=list(unet_feature_channels or ["x", "y"]),
        model_cfg=cfg_local,
        seed=int(seed),
    )
```

### checkpoint meta
```json
{
  "model_type": "coord_mlp_fourier",
  "input_dim": 12,
  "grid_shape": [64, 64],
  "out_channels": 4,
  "output_keys": ["ne", "ni", "Te", "phi"],
  "input_feature_channels": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
  "model_cfg": {...}
}
```

`models/checkpoint.py` の公開 API に save/load 対応を追加する。

## 7. `model_dispatch.py` 実装

## 7.1 分岐

次の group に入れる。
```python
elif model_name in {"unet", "fno", "ffno", "coord_mlp_fourier", "coord_mlp_siren"}:
```

## 7.2 feature contract

experimental だが、v1 では次を必須にする。
- `input_features.mode=geom_feature_pack`
- `require_pack=error`
- channel 既定は `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any`

### 理由
- `coord_feature_pack` を使うのがこのモデルの目的だから
- `legacy_xy` fallback を許すと `global_mlp` と差が曖昧になるから

## 7.3 strict validation

mainline 用の strict validator はまだ不要。
ただし experimental validator として次は落とす。
- `target_family != allvars`
- `target_vars != output_layout.vars`
- `input_features.mode != geom_feature_pack`
- feature list に重複がある

## 8. YAML 断片仕様

### Fourier 版
`configs/experimental/same_fidelity_low_data/coord_mlp_fourier_template.yaml`

```yaml
model:
  name: coord_mlp_fourier

train:
  coord_mlp_fourier:
    target_family: allvars
    target_vars: [<set-to-output_layout.vars>]
    epochs: 120
    optimizer:
      type: adamw
      schedule: cosine
    input_features:
      mode: geom_feature_pack
      require_pack: error
      features: [x, y, mask_plasma, distance_signed, distance_any]
      distance_transform:
        mode: bounded_auto
    model_cfg:
      cond_hidden: [128, 128]
      latent_dim: 128
      decoder_hidden: [256, 256, 256]
      decoder_activation: gelu
      embedding:
        type: fourier
        n_frequencies: 8
        include_raw: true
        frequency_scale: 10.0
```

### SIREN 版
`configs/experimental/same_fidelity_low_data/coord_mlp_siren_template.yaml`

```yaml
model:
  name: coord_mlp_siren

train:
  coord_mlp_siren:
    target_family: allvars
    target_vars: [<set-to-output_layout.vars>]
    epochs: 120
    optimizer:
      type: adamw
      schedule: cosine
    input_features:
      mode: geom_feature_pack
      require_pack: error
      features: [x, y, mask_plasma, distance_signed, distance_any]
      distance_transform:
        mode: bounded_auto
    model_cfg:
      cond_hidden: [128, 128]
      latent_dim: 128
      decoder_hidden: [256, 256, 256]
      siren:
        enabled: true
        w0_initial: 30.0
        w0_hidden: 1.0
```

## 9. テスト追加

### unit
- `tests/unit/models/test_coord_mlp_model.py`
  - Fourier build
  - SIREN build
  - static feature shape validation
  - forward shape
  - checkpoint roundtrip

- `tests/unit/train/test_model_dispatch_coord_mlp.py`
  - feature contract validation
  - `geom_feature_pack` がないと落ちる

### integration
- `tests/integration/test_train_coord_mlp_smoke.py`
- `tests/integration/test_infer_coord_mlp_smoke.py`

## 10. benchmark fixture 名

experimental fixture:
- `tests/fixtures/benchmark_periodic_real_m7_coord_mlp_fourier_experimental.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_coord_mlp_siren_experimental.yaml`

mainline profile lock にはまだ入れない。

## 11. 受け入れ条件

- `global_mlp` を壊さない
- Fourier 版の build/load/train/infer smoke が通る
- SIREN 版は Fourier 版 green 後に追加する
- target 名固定コードを増やさない
- feature pack 順序が train / infer で一致する
