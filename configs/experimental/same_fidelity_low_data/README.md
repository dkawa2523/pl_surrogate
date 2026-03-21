# Same-Fidelity Low-Data YAML Fragments

このディレクトリの YAML は **フル設定ではなく断片**です。

使い方:
1. 既存の `tests/fixtures/*.yaml` または自前の pipeline / benchmark YAML をベースにする
2. 対象 model の `train:` ブロックと `model.name` をこの断片で置き換える
3. `target_vars` は必ず **その run の `output_layout.vars` と同順**に置き換える
4. `tests/fixtures` にはテスト用コピーだけ置き、運用正本はこの `configs/` 側に残す

注意:
- ここでは target 名を固定しない
- `[<set-to-output_layout.vars>]` は placeholder
- `geom_feature_pack` を使うモデルでは、対応する preprocess artifact が存在する前提
