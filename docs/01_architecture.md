# 01 Architecture

この repo は、dataset、preprocessing artifact、model capability、inference result を接続する pipeline 基盤である。設計の正本は `docs/00_product_foundation_policy.md` とする。

## Pipeline

- `cleanse`: raw table / field / structure data を検証する。
- `feature`: geometry / coordinate feature pack を作る。
- `preprocess`: split、schema、scaler、target 順序、feature 順序を固定する。
- `train`: preprocessing artifact と model capability に従って学習する。
- `infer`: checkpoint と preprocessing artifact から field、QoI、diagnostics を作る。
- `evaluate`: active target に応じた metric を作る。
- `benchmark`: 複数 run の resolved config、leaderboard、selection metric を残す。

CLI は実行入口であり、contract の正本ではない。意味は dataset / preprocessing / train / infer / benchmark 側で解決する。

## Artifact Flow

下流 stage は前段が生成した artifact を読む。train / infer / benchmark が同じ artifact を読めない構成は製品 contract として扱わない。

- target order: `preprocessing/schema/output_layout.json`
- target role metadata: `preprocessing/schema/target_role_schema.json`
- condition schema: `preprocessing/schema/cond_schema.json`
- feature metadata: `preprocessing/features/coord_feature_pack_meta.json`
- channel map: `preprocessing/schema/channel_map.json`
- target transform: `preprocessing/scalers/y_scalers.json`
- runtime schema hash: `preprocessing/validation/runtime_schema_hashes.json`

## Truth Sources

- target 定義: `dataset.targets[]`
- target 順序: `output_layout.vars`
- target role: `target_role_schema.json`
- feature 順序: `coord_feature_pack_meta.json` / `channel_map.json`
- model capability: `src/plasma_surrogate/core/model_specs.py`
- benchmark primary metric: `surrogate_quality_score`
- inference objective: `inference.optimize.objective`

Benchmark profile、plot script、生成 report は target 名や出力順序の正本ではない。

## Model Lane

モデルの違いは `model_specs.py` の capability と最小 adapter で表す。新モデル追加時は `model_specs.py` から始め、train / infer / benchmark に同じ model-specific 分岐を並列に増やさない。

- `table_only`: 条件テーブルだけを入力にする。
- `table_plus_structure`: 条件テーブルに structure / grid / coordinate feature を加える。

## Out Of Scope For Architecture Docs

- dataset-specific benchmark 説明。
- generated report の inventory。
- 実験履歴や引き継ぎメモ。
- 未実装機能の利用手順。
