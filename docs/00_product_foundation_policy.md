# 00 Product Foundation Policy

この文書は `pl_surrogate` の製品基盤として守る契約を定義する。過去の経緯、fixture 固有の説明、作業メモはここに置かない。

## Purpose

条件テーブル、構造情報、grid / coordinate / graph-ready feature、target field artifact を接続し、複数モデルで surrogate 学習・推論・評価・推薦・最適化を行う基盤にする。

## Keep

- target 定義の入口は `dataset.targets[]` にする。
- target 順序の正本は `preprocessing/schema/output_layout.json` の `output_layout.vars` にする。
- target role metadata の正本は `preprocessing/schema/target_role_schema.json` にする。
- train / infer / benchmark は同じ preprocessing artifact を読む。
- feature 順序は `coord_feature_pack_meta.json` と `channel_map.json` に保存する。
- model capability は `src/plasma_surrogate/core/model_specs.py` を正本にする。
- 物理処理は target 名ではなく target role / physics symbol mapping で解決する。
- runtime input mode は `table_only` と `table_plus_structure` だけにする。
- benchmark の primary selection は `surrogate_quality_score` を使う。
- inference optimization は `inference.optimize.objective` の weighted objective contract を使う。

## Remove

- 固定 target 名を前提にした処理。
- density だけを log 特別扱いする処理。
- warning / fallback / legacy alias による互換経路。
- benchmark fixture、生成 report、作業メモを product docs に混ぜること。
- 未実装の機能を実装済み contract のように説明すること。

## Extension Order

1. target role schema validation / vocabulary stabilization
2. loss / metric protocol v2
3. role-aware output head
4. first-class CNO / UNO / UNet++ benchmark
5. graph / geometry operator lane
6. constrained multi-objective optimization

## Non-Goals

- この cleanup で Graph Neural Operator や Geo-FNO 本体は実装しない。
- 既存 benchmark を一気に壊す変更はしない。
- 後方互換 alias のために mainline code を複雑化しない。
