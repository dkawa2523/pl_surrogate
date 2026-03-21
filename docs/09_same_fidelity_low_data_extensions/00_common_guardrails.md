# 00. Common Guardrails

このファイルは、**全モデル共通の絶対条件**だけをまとめたものです。Codex は個別モデル仕様に入る前に必ず読むこと。

## 1. 壊してはいけない repo 契約

### target 契約
- target ID の唯一の真実源は `dataset.targets[].id`
- target 順序の唯一の真実源は `output_layout.vars`
- target transform の唯一の真実源は `preprocessing.scalers.target_transforms.<var>`

### feature 契約
- `coord_feature_pack` は preprocess で固める downstream artifact
- 代表チャネルは `x`, `y`, `mask_plasma`, `distance_signed`, `distance_any`
- `coord_feature_pack` を使うモデルは、**train / infer / benchmark で同じチャネル順を使う**

### strict 契約
- strict validation の入口は `src/plasma_surrogate/train/model_dispatch.py`
- mainline に昇格させるモデルは、strict validation をこのファイルへ明示的に追加する

### benchmark / compare 契約
- benchmark は dynamic target 列で集計する
- compare も固定列ではなく dynamic key を保持する
- `global_reference_mode=frozen` を使う場合でも、`model_id=global_mlp` 行の扱い以外を壊さない

## 2. 既存 repo の接続点

モデル追加時の基本変更点:
- `src/plasma_surrogate/models/*`
- `src/plasma_surrogate/models/mlp/io.py`
- `src/plasma_surrogate/train/model_dispatch.py`
- 必要なら `src/plasma_surrogate/train/trainer.py`
- 必要なら `src/plasma_surrogate/train/torch_trainer.py`
- benchmark に載せるなら `src/plasma_surrogate/benchmark/profiles.py`
- compare / fixture を増やすなら `tests/fixtures/*` と `scripts/compare_selected_models.py` の整合確認

## 3. してはいけないこと

- `ne / ni / Te / phi` を前提に分岐する
- `output_layout.vars` 以外の順序で output head を組む
- feature 順序を train と infer で変える
- compare / benchmark に固定列前提の if 文を増やす
- `tests/fixtures` を運用正本の置き場にする
- 既存 `unet`, `fno`, `deeponet_plasma` の動作を壊す大規模 refactor を最初の PR で行う

## 4. 実装時の命名規約

### model_id
- `unetpp`
- `unetpp_attn`
- `ffno`
- `coord_mlp_fourier`
- `coord_mlp_siren`
- `deeponet_pod`

### checkpoint meta `model_type`
以下のどちらかで統一する:
- `model_type == model_id`
- あるいは family を含めた `unetpp`, `ffno`, `coord_mlp_torch`, `deeponet_pod_torch`

**同じ model_id と checkpoint meta が対応することを必須**とする。

## 5. 追加モデルの最小 API

新しい torch 系モデルは、少なくとも次を持つこと。

```python
set_static_spatial_features(self, spatial_features)
forward(self, cond, training=False, spatial_features=None) -> np.ndarray
predict_fields(self, cond, spatial_features=None) -> dict[str, np.ndarray]
backward_raw(self, grad_raw, *, lr: float, weight_decay: float = 0.0, apply_step: bool = True) -> dict[str, float]
state_dict_numpy(self) -> dict[str, np.ndarray]
load_state_dict_numpy(self, state: dict[str, np.ndarray]) -> None
```

必要に応じて追加:
- `forward_features`
- `to_meta`

## 6. 実装時の段階ルール

### Step A
新モデルクラスを追加する。

### Step B
`models/mlp/io.py` に `build_model_from_name` / save / load を追加する。

### Step C
`train/model_dispatch.py` に学習分岐と strict validation を追加する。

### Step D
unit test と integration smoke を追加する。

### Step E
benchmark fixture を増やす。

## 7. mainline に昇格させる条件

次を全部満たすまで、experimental 扱いを維持する。

- target 非固定契約に従う
- preprocess artifact に依存する feature 契約を一貫して持つ
- compare / benchmark が dynamic target のまま解釈できる
- strict validation を `model_dispatch.py` に置ける
- build / load / train / infer / benchmark / compare のテストが通る

## 8. まずやるべき軽い整理

Codex には、最初の実装で次の軽い整理を推奨する。

1. `train/model_dispatch.py` 内の **unet-like torch branch** を helper 化する
   - feature pack 解決
   - `require_pack` 検証
   - distance transform 適用
   - coord feature scaling 適用

2. `models/mlp/io.py` で non-DeepONet torch wrapper の save/load を family ごとに分ける

ただし、既存挙動を変える refactor は避け、**新モデル追加に必要な最小抽出**だけを行う。
