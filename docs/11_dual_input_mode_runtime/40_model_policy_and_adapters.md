# 40. Model Policy and Adapters

## model × mode の基本方針

| model | table_only | table_plus_structure | adapter |
|---|---:|---:|---|
| global_mlp | yes | no | none |
| deeponet_pod | yes | yes | none / descriptor_branch / hybrid_pack_descriptor |
| unet | no | yes | grid_pack |
| unetpp | no | yes | grid_pack |
| unetpp_attn | no | yes | grid_pack |
| fno | no | yes | grid_pack |
| ffno | no | yes | grid_pack |
| coord_mlp_fourier | no | yes | coord_pack / hybrid_pack_descriptor |
| coord_mlp_siren | no | yes | coord_pack / hybrid_pack_descriptor |
| u_no (future) | no | yes | grid_pack |
| cno (future) | no | yes | grid_pack |
| geom_deeponet_siren (future) | no | yes | hybrid_pack_descriptor |

## 重要な原則

- model が mode を silent ignore してはいけない
- `table_only` の baseline と `table_plus_structure` の model を同一 run で混ぜない
- `global_mlp` を structure mode で無理に使わない
- `deeponet_pod` は dual-mode に向くが、adapter mode を必ず明示または自動解決する

## mainline / experimental の扱い

### mainline として維持するもの
- `global_mlp`
- `unet`
- `fno`
- `deeponet_plasma`

### same-fidelity extension で追加済み
- `unetpp`
- `unetpp_attn`
- `ffno`
- `coord_mlp_fourier`
- `coord_mlp_siren`
- `deeponet_pod`

### dual-mode での整理
- `global_mlp` は `table_only` 基準線
- `ffno`, `coord_mlp_*`, `unetpp_attn` は `table_plus_structure` の主力
- `deeponet_pod` は両 mode をまたぐ reduced-order 架橋モデル

## `deeponet_pod` の扱い

### table_only
- cond-only branch
- descriptor / latent なし
- basis は train split から fit する

### table_plus_structure
- descriptor profile が有効なら branch に加える
- latent profile が有効なら追加で branch へ結合してよい
- v1 では trunk 側に raw structure pack を渡さない

## validator 追加方針

### `src/plasma_surrogate/core/model_input_policy.py`
追加する helper:
- `resolve_supported_input_modes(model_name)`
- `resolve_allowed_adapter_modes(model_name)`
- `validate_model_input_mode(model_name, input_mode)`
- `validate_adapter_mode(model_name, input_mode, adapter_mode)`

### `train/model_dispatch.py`
既存 validator に加えて:
- `validate_runtime_input_mode(...)`
- `validate_structure_mode_requirements(...)`
- `validate_model_policy_against_mode(...)`

既存 mainline validator は壊さず、前段に mode validator を置く。
