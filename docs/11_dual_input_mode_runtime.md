# 11. Dual Input Mode Runtime

この章は、`pl_surrogate` を **入力がベクトルデータ（テーブルデータ）のみ**の場合と、**ベクトルデータ＋構造データ**の場合で、学習・推論の動線を明確に分けるための仕様です。

今回の目的は 2 つです。

1. ユーザーが **「どの mode を選ぶべきか」** を迷わず判断できるようにすること  
2. 実装側が **モデル差と特徴量差を分離**し、Codex でも安全に段階実装できるようにすること

## 今回導入する mode

- `table_only`
  - 入力は cond / process parameters などのベクトルデータのみ
  - geometry は **固定の出力文脈**としてだけ使ってよい
  - geometry をモデル入力へ混ぜない
  - geometry 最適化は行わない

- `table_plus_structure`
  - 入力はベクトルデータに加えて構造データを使う
  - 構造データは `Geometry Core` から `feature profile` として生成する
  - 必要なら `geom_space` による形状最適化を行う

## 重要な設計方針

- `target` 名は固定しない
- `dataset.targets[].id` が target 定義の真実源
- `output_layout.vars` が target 順序の真実源
- `preprocessing.scalers.target_transforms.<var>` が前処理契約の真実源
- `runtime.input_mode` を **train / infer / benchmark の第一級ルーティングキー** にする
- モデル名で feature profile を増殖させず、**mode と profile で切る**
- `table_only` と `table_plus_structure` の結果は benchmark / compare でも混ぜない

## 読み順

1. `11_dual_input_mode_runtime/README.md`
2. `11_dual_input_mode_runtime/00_user_view.md`
3. `11_dual_input_mode_runtime/10_architecture_and_contract.md`
4. `11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md`
5. `11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md`
6. `11_dual_input_mode_runtime/40_model_policy_and_adapters.md`
7. `11_dual_input_mode_runtime/50_geom_space_and_optimization.md`
8. `11_dual_input_mode_runtime/60_yaml_templates_and_examples.md`
9. `11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md`
10. `11_dual_input_mode_runtime/80_migration_notes.md`

## この章で追加する主な責務

- `runtime.input_mode` の正規化と fail-fast
- `structure.feature_profile / descriptor_profile / latent_profile` の mode 整合性チェック
- `table_only` では geometry は output context のみ、`table_plus_structure` では model input として扱う、という明確な分離
- benchmark / compare / checkpoint metadata への `input_mode_effective` の埋め込み

## 典型的な使い分け

### `table_only`
- geometry がケース間で変化しない
- まず cond-only baseline をきれいに回したい
- 既存 `global_mlp` や reduced-order cond-only 系を使いたい

### `table_plus_structure`
- 構造差が出力に効く
- SDF / part structure / descriptor を入力へ入れたい
- 形状も含めて最適化したい

## 非目標

- raw mask の画素値を直接最適化する
- point-cloud / mesh / transformer 系へ一気に全面移行する
- `table_only` なのに暗黙に structure features を混ぜる
- `table_plus_structure` なのに geometry を silently ignore する
