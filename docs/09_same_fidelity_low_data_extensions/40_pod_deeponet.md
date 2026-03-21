# 40. POD-DeepONet 実装仕様

## 1. 目的

少量データ向けに、full-field を直接学ぶ代わりに **低ランク basis 上の係数**を学ぶ reduced-order operator を追加する。

追加 model_id:
- `deeponet_pod`

このモデルは **experimental** とする。

## 2. 重要方針

v1 では、既存 `deeponet_plasma` の mainline strict contract に無理に入れない。

代わりに:
- **train split 上で POD basis を fit**
- その basis を checkpoint に埋め込む
- infer 時に新しい preprocess artifact を要求しない

この方針により、実装差分を抑えつつ reduced-order の効果を確認できる。

## 3. 変更対象ファイル

### 新規作成
- `src/plasma_surrogate/models/deeponet/pod_deeponet_torch.py`

### 変更
- `src/plasma_surrogate/models/mlp/io.py`
- `src/plasma_surrogate/train/model_dispatch.py`
- unit / integration tests

## 4. v1 の設計

## 4.1 位置づけ

v1 の `deeponet_pod` は、実装上は「固定 POD basis を trunk とみなす reduced-order DeepONet」とする。

### 入力
- branch: 条件ベクトル `cond`
- trunk: **固定された POD basis**

### 出力
- 各 target ごとの POD coefficient
- basis 再構成で full-field を得る

## 4.2 basis fitting 方針

### fit 対象
- `ctx.y_scaled[ctx.tr]`
- つまり **train split の scaled target** のみ

### basis fitting 単位
v1 は target ごとに独立 basis を持つ。

```text
for var in y_vars:
    snapshots = y_scaled_train[:, var_index, :, :]
    flatten -> [N, H*W]
    mean subtract
    SVD
    keep top-r
```

保存するもの:
- `basis[var]`: `[rank, H, W]`
- `mean[var]`: `[H, W]`
- `rank[var]`

### 既定 rank
- `rank: 32`
- ただし `H*W` と sample 数に応じて自動で clamp する

## 4.3 wrapper API

`PODDeepONetTorch` は、torch-style full-field wrapper にする。

```python
class PODDeepONetTorch:
    def __init__(
        self,
        *,
        input_dim: int,
        grid_shape: tuple[int, int],
        out_channels: int,
        output_keys: list[str] | None,
        pod_basis: dict[str, np.ndarray],
        pod_mean: dict[str, np.ndarray],
        model_cfg: dict[str, Any],
        seed: int,
    ) -> None: ...
```

必須メソッド:
- `forward`
- `predict_fields`
- `backward_raw`
- `state_dict_numpy`
- `load_state_dict_numpy`
- `to_meta`（推奨）

## 4.4 ネットワーク本体

### branch network
- 入力: `cond`
- hidden: 2〜3 層
- 出力: `sum(rank[var] for var in y_vars)` 個の coefficient

### reconstruction
```text
for each var:
    coeffs[var] @ basis[var] + mean[var]
```

最終 shape:
- `[B, C_out, H, W]`

## 5. `model_dispatch.py` 実装

## 5.1 分岐

`deeponet_pod` は専用分岐にする。

理由:
- builder に渡す前に `ctx.y_scaled[ctx.tr]` から basis を fit したい
- `deeponet_plasma` の plain mainline contract と分離したい

擬似コード:
```python
elif model_name == "deeponet_pod":
    cfg = dict(train_cfg.get("deeponet_pod", {}))
    target_vars = resolve_allvars(...)
    target_indices = [ctx.y_vars.index(v) for v in target_vars]
    y_train = ctx.y_scaled[ctx.tr][:, target_indices, :, :]
    pod_basis, pod_mean = fit_pod_basis(...)
    model_cfg = dict(cfg.get("model_cfg", {}))
    model_cfg["__pod_basis__"] = pod_basis
    model_cfg["__pod_mean__"] = pod_mean
    model = build_model_from_name(... model_name="deeponet_pod", model_cfg=model_cfg ...)
    # training は torch-style full-field branch で実施
```

## 5.2 validation

experimental validator として次を入れる。
- `target_family` は `allvars` のみ
- `target_vars == output_layout.vars`
- `model_cfg.basis.rank >= 1`
- `fit_scope=train_only` 以外は落とす

## 6. `models/mlp/io.py` 実装

### builder

```python
if name == "deeponet_pod":
    cfg_local = dict(cfg)
    pod_basis = cfg_local.pop("__pod_basis__")
    pod_mean = cfg_local.pop("__pod_mean__")
    return PODDeepONetTorch(
        input_dim=int(input_dim),
        grid_shape=tuple(grid_shape),
        out_channels=int(out_channels),
        output_keys=output_keys,
        pod_basis=pod_basis,
        pod_mean=pod_mean,
        model_cfg=cfg_local,
        seed=int(seed),
    )
```

### checkpoint
basis と mean は checkpoint に埋め込む。

#### meta 例
```json
{
  "model_type": "deeponet_pod",
  "input_dim": 12,
  "grid_shape": [64, 64],
  "out_channels": 4,
  "output_keys": ["ne", "ni", "Te", "phi"],
  "model_cfg": {
    "hidden_dim": 128,
    "basis": {"rank": 32, "fit_scope": "train_only", "per_var": true}
  },
  "basis_keys": ["ne", "ni", "Te", "phi"],
  "basis_rank_by_var": {"ne": 32, "ni": 32, "Te": 32, "phi": 32}
}
```

#### weights
`state_dict_numpy()` に次を含める:
- `branch::...` の network weights
- `basis::<var>`
- `mean::<var>`

## 7. YAML 断片仕様

`configs/experimental/same_fidelity_low_data/deeponet_pod_template.yaml`

```yaml
model:
  name: deeponet_pod

train:
  deeponet_pod:
    target_family: allvars
    target_vars: [<set-to-output_layout.vars>]
    epochs: 120
    optimizer:
      type: adamw
      schedule: cosine
    model_cfg:
      hidden_dim: 128
      latent_dim: 128
      basis:
        rank: 32
        fit_scope: train_only
        per_var: true
        center: true
    selection:
      mode: best_val_allvars_balance
```

## 8. テスト追加

### unit
- `tests/unit/models/test_pod_deeponet_model.py`
  - basis fit helper
  - forward / reconstruction shape
  - checkpoint roundtrip

- `tests/unit/train/test_model_dispatch_pod_deeponet.py`
  - train split only で basis fit される
  - invalid rank / invalid fit_scope を落とす

### integration
- `tests/integration/test_train_pod_deeponet_smoke.py`
- `tests/integration/test_infer_pod_deeponet_smoke.py`

## 9. benchmark fixture 名

experimental fixture:
- `tests/fixtures/benchmark_periodic_real_m7_deeponet_pod_experimental.yaml`

既存 mainline profile にはまだ追加しない。

## 10. 受け入れ条件

- `deeponet_plasma` を壊さない
- basis は train split のみで fit される
- checkpoint だけで infer できる
- target 名固定コードを増やさない
- build/load/train/infer smoke が通る
