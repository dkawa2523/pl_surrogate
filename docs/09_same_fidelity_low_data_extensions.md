# 09. Same-Fidelity Low-Data Extensions

この章は、**同一 fidelity・physics-informed なし**の前提で、既存 `pl_surrogate` に少量データ向けモデル拡張を追加するための実装仕様です。

この章の目的は、VS Code の Codex Chat が workspace 内のドキュメントを順に読み、**repo の契約を壊さずに段階実装できる状態**を作ることです。

## 対象モデル

第1優先:
- `unetpp`
- `ffno`

第2優先:
- `unetpp_attn`
- `coord_mlp_fourier`
- `coord_mlp_siren`

第3優先:
- `deeponet_pod`

## 先に読む順番

1. `docs/09_same_fidelity_low_data_extensions/README.md`
2. `docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md`
3. 実装したいモデルの個別仕様
4. `docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md`

## 個別仕様ファイル

- `10_unetpp_and_attention.md`
- `20_ffno.md`
- `30_coord_mlp_family.md`
- `40_pod_deeponet.md`

## 配置方針

- **仕様書**は `docs/` 配下に置く。
- **モデル別の YAML 断片**は `configs/experimental/same_fidelity_low_data/` に置く。
- **テスト用の fixture** は `tests/fixtures/` に追加するが、運用設定の正本にはしない。

## この章のルール

- target 名を固定しない。
- target 順序は常に `output_layout.vars` を真実源にする。
- `coord_feature_pack` を使うモデルでは、train / infer / benchmark で feature 順序をずらさない。
- `tests/fixtures` はテンプレ扱いとし、実運用向け YAML は `configs/` に置く。

## 実装順

### Phase 1
- `unetpp`
- `ffno`

### Phase 2
- `unetpp_attn`
- `coord_mlp_fourier`
- `coord_mlp_siren`

### Phase 3
- `deeponet_pod`

## 非目標

この章では次を扱わない:
- PINO
- physics-informed loss
- multi-fidelity
- active learning
- Geo-FNO / mesh / graph 系

## Codex への渡し方

最初の指示は次の形にする。

```text
Read these files in order:
1) docs/09_same_fidelity_low_data_extensions.md
2) docs/09_same_fidelity_low_data_extensions/README.md
3) docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md
4) docs/09_same_fidelity_low_data_extensions/10_unetpp_and_attention.md
5) docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md

Implement Phase 1 only.
Do not introduce physics-informed changes.
Do not hardcode target names.
Keep benchmark and compare dynamic-target compatible.
```
