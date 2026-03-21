# 10. UNet++ / Attention-UNet++ 実装仕様

## 1. 目的

`unet` の feature 契約と target 契約を保ったまま、**少量データでの局所構造再現性**を改善する。

今回の実装では次を追加する。
- `unetpp`
- `unetpp_attn`

## 2. 実装優先順

### Phase 1a
- `unetpp`
- deep supervision なし
- attention なし

### Phase 1b
- `unetpp` に optional deep supervision を追加

### Phase 2
- `unetpp_attn`
- skip connection 上に attention gate を追加

## 3. mainline-candidate として守る条件

`unetpp` は既存 `unet` と同じ外部契約を守る。

必須:
- `train.unetpp.target_family=allvars`
- `train.unetpp.target_vars == output_layout.vars`
- `train.unetpp.input_features.mode=geom_feature_pack`
- `train.unetpp.model_cfg.output_heads.mode=shared`
- `train.unetpp.selection.mode=best_val_allvars_balance`
- feature channel は `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any` を既定にする

`unetpp_attn` も同じ契約を守る。違いは内部 backbone のみ。

## 4. 変更対象ファイル

### 新規作成
- `src/plasma_surrogate/models/unet/unetpp.py`

### 変更
- `src/plasma_surrogate/models/mlp/io.py`
- `src/plasma_surrogate/train/model_dispatch.py`
- `tests/unit/models/` 配下の該当テスト
- `tests/integration/` 配下の smoke test
- 必要なら `src/plasma_surrogate/benchmark/profiles.py`

## 5. モデル設計

## 5.1 外部 API

`UNetPPBaseline` は、`UNetBaseline` と同じ wrapper API を持つこと。

```python
class UNetPPBaseline:
    def __init__(
        self,
        *,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int,
        output_keys: list[str] | None,
        with_rho_eff_head: bool,
        head_mlp: dict[str, Any],
        backend: str,
        input_feature_channels: list[str],
        conv_cfg: dict[str, Any],
        output_heads: dict[str, Any],
        seed: int,
    ) -> None: ...
```

必要メソッド:
- `set_static_spatial_features`
- `_resolve_spatial_features`
- `_feature_map`
- `forward`
- `predict_fields`
- `backward_raw`
- `state_dict_numpy`
- `load_state_dict_numpy`

## 5.2 最小構成

v1 は **depth=2 の nested skip** だけでよい。複雑にしすぎない。

推奨ノード:
- `x00`, `x10`, `x20`
- `x01`, `x11`
- `x02`

出力は `x02` のみ。

### なぜ depth=2 から始めるか
- 現行 `simple_unet.py` も depth 1/2 の構成を中心にしている
- smoke と checkpoint 実装を小さく保てる
- まず既存 `unet` と比較可能な差分にするため

## 5.3 deep supervision の扱い

Phase 1a では不要。

Phase 1b で追加する場合のみ:
- `conv_cfg.deep_supervision.enabled: bool`
- `conv_cfg.deep_supervision.aux_weights: list[float]`

実装ルール:
- 推論時は final head のみ使う
- 学習時のみ aux head を supervised loss に混ぜる
- aux loss の既定は final 1.0, aux 0.3
- trainer 側の変更が大きいなら、Phase 1 の PR では入れない

## 5.4 attention gate の扱い

`unetpp_attn` では skip tensor に対して gate を入れる。

推奨:
- 各 nested skip の concat 前に gate
- bottleneck attention は後回し

設定例:
```yaml
attention_cfg:
  enabled: true
  reduction: 2
  gate_activation: sigmoid
```

## 6. builder / checkpoint 実装

## 6.1 `models/mlp/io.py`

### `build_model_from_name` に追加

```python
if name == "unetpp":
    return UNetPPBaseline(...)

if name == "unetpp_attn":
    cfg_local = dict(cfg)
    conv_cfg = dict(cfg_local.get("conv_cfg", {}))
    attention_cfg = dict(conv_cfg.get("attention_cfg", {}))
    attention_cfg["enabled"] = True
    conv_cfg["attention_cfg"] = attention_cfg
    cfg_local["conv_cfg"] = conv_cfg
    return UNetPPBaseline(... cfg=cfg_local ...)
```

### checkpoint meta
必要な meta 例:
```json
{
  "model_type": "unetpp",
  "input_dim": 12,
  "grid_shape": [64, 64],
  "out_channels": 4,
  "output_keys": ["ne", "ni", "Te", "phi"],
  "backend": "torch",
  "input_feature_channels": ["x", "y", "mask_plasma", "distance_signed", "distance_any"],
  "with_rho_eff_head": false,
  "head_mlp": {},
  "conv_cfg": {...},
  "output_heads": {"mode": "shared"}
}
```

`load_mlp_checkpoint` に `model_type in {"unetpp", "unetpp_attn"}` を追加する。

## 7. `model_dispatch.py` 実装

`unetpp` / `unetpp_attn` は **unet-like torch branch** で処理する。

### 追加方針
以下の分岐へ組み込む:
```python
elif model_name in {"unet", "fno", "unetpp", "unetpp_attn", ...}:
```

### strict validation
`_validate_unet_like_mainline_contract(...)` を再利用できるなら再利用する。
必要なら `model_name in {"unet", "unetpp", "unetpp_attn"}` で shared output head を必須化する。

### 注意点
- `input_features.mode=geom_feature_pack` を mainline-candidate では必須にする
- `require_pack` の `off / warn / error` 契約を `unet` と同じにする
- distance transform と coord feature scaling も `unet` と同じ helper を使う

## 8. YAML 断片仕様

`configs/experimental/same_fidelity_low_data/unetpp_template.yaml` を参照。

最低限の shape:
```yaml
model:
  name: unetpp

train:
  unetpp:
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
      backend: torch
      conv_cfg:
        base_channels: 32
        depth: 2
        upsample_mode: bilinear
        nested_skip: true
        deep_supervision:
          enabled: false
      output_heads:
        mode: shared
    selection:
      mode: best_val_allvars_balance
```

## 9. テスト追加

### unit
- `tests/unit/models/test_unetpp_model.py`
  - build できる
  - `set_static_spatial_features` shape validation
  - forward 出力 shape
  - checkpoint save/load roundtrip

- `tests/unit/train/test_model_dispatch_unetpp.py`
  - strict contract validation
  - `input_features.mode=geom_feature_pack` で channel 検証

### integration
- `tests/integration/test_train_unetpp_smoke.py`
- `tests/integration/test_cli_unetpp_smoke.py`
- `tests/integration/test_infer_unetpp_smoke.py`

## 10. benchmark fixture 名

mainline-candidate として試す場合:
- `tests/fixtures/benchmark_periodic_real_m7_unetpp_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_unetpp_attn_isolated_mainline.yaml`

ただし `unetpp_attn` は `unetpp` が green になってから追加。

## 11. 受け入れ条件

`unetpp` 実装完了条件:
- 既存 `unet` のテストが落ちない
- `unetpp` の build/load/train/infer smoke が通る
- `target_vars == output_layout.vars` を守る
- `geom_feature_pack` を train と infer で同一順序で使う
- benchmark / compare が dynamic target のまま動く

`unetpp_attn` 追加条件:
- `unetpp` green 後
- attention を無効にしたとき `unetpp` と同等の shape / API を保つ
