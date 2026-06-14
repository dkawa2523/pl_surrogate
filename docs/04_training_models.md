# 04 Training And Models

training は preprocessing artifact と `src/plasma_surrogate/core/model_specs.py` の capability に従う。新モデル追加は必ず `model_specs.py` から始める。

## Training Contract

- target 定義: `dataset.targets[]`
- target 順序: `preprocessing/schema/output_layout.json` の `output_layout.vars`
- target role: `preprocessing/schema/target_role_schema.json`
- feature 順序: `coord_feature_pack_meta.json` / `channel_map.json`
- model capability: `src/plasma_surrogate/core/model_specs.py`
- runtime metadata: checkpoint / inference / benchmark で同じ required keys を検証する

checkpoint と runtime request の schema hash または input mode が一致しない場合は fail-fast とする。

## First-Class Models

`model_specs.py` の `product_status` と `product_category` がこの分類の正本である。

| category | models |
| --- | --- |
| baseline | `global_mlp` |
| grid local | `unet`, `unetpp`, `unetpp_attn` |
| spectral / operator | `fno`, `ffno`, `u_no`, `cno` |
| coordinate / operator | `deeponet_plasma`, `coord_mlp_siren`, `geom_deeponet_siren` |

Factory は `get_model_spec(model_name)` から family builder に dispatch する。train / infer / benchmark に model capability の重複定義を増やさない。

## Experimental / Archive

次の model は実装が残るが product first-class ではない。

- `deeponet_pod`
- `deeponet_plasma_pod`
- `geom_deeponet_pod`
- `coord_mlp_fourier`
- `coord_mlp_pod_residual`
- `unet_operator_v2`
- `cno_operator_unet`

使う場合は local experiment として扱い、製品 docs の推奨 path にはしない。

## Loss Protocol

製品向けの第一選択は protocol 指定だけにする。

```yaml
train:
  loss:
    protocol: plasma_surrogate_v2
```

この protocol は `huber`, `plasma_only`, `region_balance`, multiscale `spatial_consistency` を標準化する。`target_role_schema.json` に `positive: true` がある target には role-aware positive penalty を付ける。role schema がない場合は target 名を推定せず、全 target 共通 default に留める。

## Output Heads

first-class model の product examples は `output_heads.mode: shared` を使う。target group 別 head が必要になった場合は、target role を使って別途設計する。

## Physics Training

physics-aware training は `physics.symbols` または `target_role_schema.json` の一意な role / field_family で target を解決する。解決できない場合は fail-fast とする。
