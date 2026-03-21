# Same-Fidelity Low-Data Extensions: Codex 実装ガイド

このディレクトリは、Codex Chat がそのまま読み込んで実装できるように、**実装順・守る契約・変更対象ファイル・テスト観点**を分割した仕様書です。

## この仕様の前提

- 既存 repo の中心モデルは `global_mlp`, `unet`, `fno`, `deeponet_plasma`。
- `global_mlp` は条件ベクトル中心、`unet` / `fno` / `deeponet_plasma` は geometry / coord feature を直接使う。
- strict contract の入口は `src/plasma_surrogate/train/model_dispatch.py`。
- モデル追加時の接続点は `src/plasma_surrogate/models/*`, `src/plasma_surrogate/models/mlp/io.py`, `src/plasma_surrogate/train/model_dispatch.py` を基本とする。

## Codex が最初に理解すべきこと

1. **truth source を壊さないこと**
   - target ID の真実源: `dataset.targets[].id`
   - target 順序の真実源: `output_layout.vars`
   - target transform の真実源: `preprocessing.scalers.target_transforms.<var>`

2. **feature contract を壊さないこと**
   - `coord_feature_pack` は preprocess で固めた downstream 用 artifact。
   - `unet`, `fno`, `deeponet_plasma` はこれを強く前提としている。

3. **dynamic metrics を壊さないこと**
   - benchmark と compare は固定列ではなく dynamic target ベースで動く。
   - 新モデル追加時も固定的な `phi` 列などを前提にしない。

## 実装優先順位

### Phase 1: mainline 候補
- `unetpp`
- `ffno`

### Phase 2: experimental 候補
- `unetpp_attn`
- `coord_mlp_fourier`
- `coord_mlp_siren`

### Phase 3: experimental 候補
- `deeponet_pod`

## 読む順番

### 全モデル共通で先に読む
1. `00_common_guardrails.md`
2. `90_benchmark_and_tests.md`

### model ごとの読み順
- U-Net 系を実装する場合: `10_unetpp_and_attention.md`
- FNO 系を実装する場合: `20_ffno.md`
- MLP 系を実装する場合: `30_coord_mlp_family.md`
- Operator / reduced-order 系を実装する場合: `40_pod_deeponet.md`

## 実装の原則

- **まず Phase 1 の smoke を通す**。
- その後に benchmark 用 fixture を増やす。
- いきなり Phase 2 / 3 を同時実装しない。
- 既存 `unet` / `fno` / `deeponet_plasma` のコードを大規模に壊す refactor を避ける。
- まずは新クラス・新 `model_name` を増やし、既存モデルは無変更で green を保つ。

## 追加する推奨 model_id

- `unetpp`
- `unetpp_attn`
- `ffno`
- `coord_mlp_fourier`
- `coord_mlp_siren`
- `deeponet_pod`

## 実装レーン

### mainline-candidate
次を満たすもの:
- `target_family=allvars`
- `target_vars == output_layout.vars`
- `input_features.mode=geom_feature_pack`（必要なモデルのみ）
- shared output head を維持できる
- benchmark / compare が dynamic target のまま解釈できる
- strict validation を `model_dispatch.py` に置ける

現時点の mainline-candidate:
- `unetpp`
- `ffno`
- `unetpp_attn`（`unetpp` 実装後に昇格判断）

### experimental
- `coord_mlp_fourier`
- `coord_mlp_siren`
- `deeponet_pod`

## Codex 用の最小プロンプト

### Phase 1
```text
Read docs/09_same_fidelity_low_data_extensions.md and the subdocs README + 00_common_guardrails + 10_unetpp_and_attention + 20_ffno + 90_benchmark_and_tests.
Implement only Phase 1.
Add new model ids unetpp and ffno.
Do not modify existing model behavior.
Keep all target handling dynamic.
Run the relevant unit and integration tests.
```

### Phase 2
```text
Read docs/09_same_fidelity_low_data_extensions/30_coord_mlp_family.md.
Implement coord_mlp_fourier first, then coord_mlp_siren.
Route them through the torch-style training path.
Keep them experimental; do not add them to mainline profile locks yet.
```

### Phase 3
```text
Read docs/09_same_fidelity_low_data_extensions/40_pod_deeponet.md.
Implement deeponet_pod as an experimental reduced-order model.
Fit POD basis on train split only.
Persist basis inside the checkpoint so infer does not depend on new preprocess artifacts in v1.
```
