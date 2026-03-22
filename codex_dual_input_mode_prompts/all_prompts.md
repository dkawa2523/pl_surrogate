# 01_phase0_input_mode_contract_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。目的は、train / infer / benchmark の入口で `table_only` と `table_plus_structure` を明確に分けることです。


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスクは **Phase 0: input mode contract の追加** だけです。
まだ preprocess の大改造、geom_space、future model 追加には着手しないでください。

実装目標:
- 新規 `runtime.input_mode` 契約を追加する
- mode 定数と正規化 helper を追加する
- checkpoint / benchmark / infer summary で使う metadata key を定義する
- 現時点では既存挙動を変えず、registry と validation の土台だけ入れる

追加するもの:
1. `src/plasma_surrogate/core/input_modes.py`
   - `TABLE_ONLY`, `TABLE_PLUS_STRUCTURE`
   - `normalize_input_mode_cfg(...)`
   - `validate_input_mode_cfg(...)`
   - metadata key helper
2. `src/plasma_surrogate/core/model_input_policy.py`
   - まずは最小版でよい
   - model ごとの supported input modes を定義する
   - adapter mode の定数を定義する
3. docs で参照される import path を壊さないよう必要最小限の `__init__` export を加える

絶対条件:
- target 名を固定しない
- target / output / transform の truth source を変えない
- silent fallback を入れない
- 既存 mainline config が読み込めなくなる変更はしない
- legacy config は migration helper で既定解決してよいが、effective metadata には最終 mode を必ず保存できるようにする

今回まだやらないこと:
- preprocess runner の structure profile 解決
- geometry provider の parametric parts
- optimize の geom_space
- benchmark の mode 分離
- model dispatch の full gating

作業完了時の出力要件:
- 変更ファイル一覧
- 追加ファイル一覧
- `table_only` / `table_plus_structure` の解決ルール
- 実行したテスト
- 次フェーズに残した TODO

```

# 02_phase1_table_only_runtime_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Phase 1a: `table_only` を first-class runtime にする** 作業だけを行います。


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- `table_only` のとき、geometry は output context としてのみ使う path を明確化する
- structure profile を使おうとしたら fail-fast させる
- preprocess / train / infer の metadata に `input_mode_effective=table_only` を通す

実装目標:
1. preprocess runner
   - `runtime.input_mode=table_only` なら structure profile 解決を行わない
   - `runtime.structure.feature_profile != none` ならエラー
   - metadata に `input_mode_effective`, `has_structure_inputs_effective=false` を保存
2. infer engine
   - `table_only` では geometry を model input に使わない
   - `geom_ref.geom_param` が来たら reject
3. workflow / CLI glue
   - run summary に `input_mode_effective` を残す
4. yaml templates
   - `configs/experimental/dual_input_modes/table_only_*.yaml` が読みやすいように最小追記

絶対条件:
- fixed geometry output context は壊さない
- `global_mlp` と `deeponet_pod` の既存 path を壊さない
- `coord_feature_pack` がなくても `table_only` で動くようにする
- `table_only` で structure-aware model を通さない

今回まだやらないこと:
- structure profile registry の本実装
- geom_space
- benchmark の mode-aware compare
- descriptor lane

作業完了時:
- どこで table_only が解決されるかを短く要約
- fail-fast ルールを列挙
- テスト結果を示す

```

# 03_phase1_structure_registry_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Phase 1b: structure registry と profile abstraction** を実装します。


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- allowed coord feature channels の重複定義をやめる
- feature profile / descriptor profile / latent profile を 1 箇所で管理する
- `table_plus_structure` の structure profile 解決を registry 経由にする

実装目標:
1. 新規 `src/plasma_surrogate/features/structure_feature_registry.py`
   - allowed spatial channels
   - `geom_v1_mainline`
   - `boundary_plus_v1`
   - `part_lite_v1`
   - `part_semantic_v1`
   - `struct_desc_v1`
   - `shape_ae_v1`
   - `part_latent_v1`
   - normalize / validate helper
2. `preprocessing/runner.py` と `infer/engine.py`
   - `_ALLOWED_COORD_FEATURE_CHANNELS` を local 定義しない
   - registry を import する
3. `table_plus_structure` で `runtime.structure.feature_profile` を必須にする
4. checkpoint metadata に `structure_feature_profile_effective` などを保存する足場を入れる

絶対条件:
- 既存 mainline 互換 profile は `geom_v1_mainline`
- mainline exact 5 channels は残す
- profile 名でチャネル集合を解決し、YAML へチャネル列を直書きしない
- `part_semantic_v1` は optional だが registry 上は予約しておく

今回まだやらないこと:
- descriptor artifact の完全実装
- provider の parametric parts
- optimize.geom_space
- benchmark row 拡張

作業完了時:
- registry に入った profile 一覧
- 既存コードの重複がどこから除去されたか
- 実行した unit test

```

# 04_phase2_preprocess_train_infer_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Phase 2a: preprocess / train / infer を dual-mode でつなぐ** 作業です。


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- `runtime.input_mode` に応じて preprocess / train / infer が一貫して動くようにする
- metadata を preprocess -> checkpoint -> infer summary へ通す
- `table_plus_structure` では feature profile を読み、`table_only` では none を通す

実装目標:
1. preprocess outputs
   - `input_mode_effective`
   - `structure_feature_profile_effective`
   - `structure_descriptor_profile_effective`
   - `structure_latent_profile_effective`
   - `has_structure_inputs_effective`
2. checkpoint meta
   - 上記 metadata を保存
3. infer engine
   - checkpoint meta と request mode を照合する
   - 矛盾時は fail-fast
4. workflows
   - infer / evaluate summary に effective mode を出す

絶対条件:
- `table_only` の runs は structure profile を none に正規化
- `table_plus_structure` は feature profile が欠けていたらエラー
- `output_layout.vars` や target transforms には触れない
- 新規 metadata key は benchmark / compare で後から参照しやすい命名にする

今回まだやらないこと:
- model policy の full validator
- benchmark compare の mode separation
- geom_space
- descriptor lane

作業完了時:
- metadata フロー図の簡易説明
- 追加した meta key 一覧
- 実行した smoke test

```

# 05_phase2_model_policy_dispatch_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Phase 2b: model policy と dispatch gating** を実装します。


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- model × input_mode × adapter_mode の整合性を train 入口で強く検証する
- `global_mlp` を structure mode で silently 使えないようにする
- `deeponet_pod` の dual-mode 許可を明示する

実装目標:
1. `src/plasma_surrogate/core/model_input_policy.py`
   - `global_mlp -> table_only`
   - `deeponet_pod -> table_only, table_plus_structure`
   - `unet`, `unetpp`, `unetpp_attn`, `fno`, `ffno`, `coord_mlp_fourier`, `coord_mlp_siren -> table_plus_structure`
   - adapter_mode support matrix
2. `train/model_dispatch.py`
   - mode validator を既存 strict validator の前段に入れる
   - `runtime.input_mode` と model policy を照合
   - `table_plus_structure` で structure-aware adapter が必要な model に `adapter_mode` を解決
3. `models/mlp/io.py`
   - checkpoint meta に `input_mode_effective`, `structure_adapter_mode_effective` を保存
4. config templates
   - `dual_input_modes/*.yaml` を docs に沿って最小更新

絶対条件:
- mainline exact 5ch validator は残す
- target 名や target 数に依存しない
- model 名ごとに新しい feature profile を増やさない
- `auto` adapter は convenience であり silent ignore ではない

今回まだやらないこと:
- benchmark compare separation
- provider geom_space
- descriptor artifact の完全実装
- future models

作業完了時:
- support matrix の実装場所
- 追加した validation rule
- failing config example を 2 つ示す

```

# 06_phase3_benchmark_compare_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Phase 3: benchmark / compare を input_mode aware にする** 作業です。


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- benchmark row に input_mode / structure profile metadata を出す
- compare で異なる input_mode を既定で混ぜない
- fixtures と smoke benchmark を dual-mode 用に整える

実装目標:
1. `benchmark/runner.py`
   - row に以下を追加:
     - `input_mode_effective`
     - `structure_feature_profile_effective`
     - `structure_descriptor_profile_effective`
     - `structure_latent_profile_effective`
     - `structure_adapter_mode_effective`
     - `geometry_provider_mode_effective`
2. `scripts/compare_selected_models.py`
   - 既定では `require_same_input_mode=true`
   - mode が違う行は compare 対象外にするか、明示 warning を出す
3. fixtures
   - table_only と table_plus_structure の最小 fixture を追加
4. tests
   - benchmark row が mode-aware かを確認
   - compare が mode mismatch を拒否することを確認

絶対条件:
- dynamic target metrics を壊さない
- global reference も mode-aware に扱う
- fixed columns を増やしすぎない
- table_only と table_plus_structure の leaderboards を暗黙に混ぜない

作業完了時:
- 追加した leaderboard columns
- mode-aware compare のルール
- 実行した benchmark smoke

```

# 07_phase4_provider_geom_space_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Phase 4: provider runtime と geom_space** を実装します。


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- geometry provider を `fixed` と `parametric_parts` で分ける
- `table_plus_structure` でのみ `geom_space` を使えるようにする
- optimize backend を cond + geom の joint 提案に拡張する

実装目標:
1. `data/geometry_provider.py`
   - `FixedGeometryProvider` の後方互換を保ちつつ、mode-aware provider へ拡張
   - `provider_mode=fixed | parametric_parts`
   - `geom_ref.geom_param` を `parametric_parts` でのみ解釈
2. `infer/optimize.py`
   - `geom_space` 引数を追加
   - random / optuna / csv backend で cond と geom を同時提案できるようにする
   - `table_only` では `geom_space` を reject
3. infer engine
   - `table_plus_structure` で provider を通して Geometry Core を再解決
4. tests
   - `table_only` で `geom_space` を reject
   - `table_plus_structure` で joint search smoke

絶対条件:
- raw pixel optimization は実装しない
- `geom_space` は意味のある low-dimensional params のみ
- backward compatibility を壊さない
- geometry provider の fixed path は引き続き動く

今回まだやらないこと:
- learned latent geometry encoder
- future models
- benchmark objective の大幅追加

作業完了時:
- `geom_space` contract を短く要約
- backend 変更点
- 実行テスト

```

# 08_phase5_descriptor_latent_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Phase 5: descriptor / latent lane の受け口** を実装します。


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- descriptor profile を artifact として扱う
- `deeponet_pod` を dual-mode reduced-order model として整える
- latent profile は v1 では metadata と hook だけを入れる

実装目標:
1. `features/structure_descriptors.py` を追加
   - `struct_desc_v1` の deterministic builder
2. preprocess
   - descriptor profile が有効なら descriptor artifact を保存
3. `deeponet_pod`
   - `table_only`: cond-only branch
   - `table_plus_structure`: descriptor_branch を通せる
   - latent profile が有効でも、v1 は external latent artifact 読み込み hook のみ
4. infer / io
   - descriptor / latent metadata を checkpoint と summary に保存
5. tests
   - table_only deeponet_pod smoke
   - table_plus_structure deeponet_pod descriptor smoke

絶対条件:
- descriptor は fixed length vector
- pairwise O(P^2) 全展開はしない
- latent encoder 学習自体は今回入れない
- `global_mlp` を shape-aware 本命にしない

作業完了時:
- descriptor artifact 形式
- deeponet_pod の dual-mode 差分
- 残した future work

```

# 09_optional_u_no_dual_mode_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Optional Phase: U-NO を dual-mode 基盤へ追加** します。

前提:
- 01〜08 が完了していること
- `table_plus_structure` の grid_pack path が安定していること


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- `u_no` を structure mode 専用の grid model として追加する
- `ffno` / `unetpp` と同じ feature profile / adapter 契約を使う
- benchmark に mode-aware に載せる

実装目標:
1. `src/plasma_surrogate/models/uno/...`
2. `models/mlp/io.py` build/load/save 追加
3. `train/model_dispatch.py` へ `u_no` integration
4. fixture / smoke test / benchmark profile 追加

絶対条件:
- `table_only` では使えない
- `adapter_mode=grid_pack`
- feature profile は registry から解決
- dynamic target 契約を壊さない

作業完了時:
- 変更ファイル一覧
- validator 追加点
- 実行テスト

```

# 10_optional_cno_dual_mode_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Optional Phase: CNO を dual-mode 基盤へ追加** します。

前提:
- 01〜08 が完了していること
- grid lane と benchmark mode separation が安定していること


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- `cno` を structure mode 専用の grid model として追加する
- U-NO と同じ runtime policy に乗せる
- profile / adapter / metadata 契約を共通化する

実装目標:
1. `src/plasma_surrogate/models/cno/...`
2. `models/mlp/io.py`
3. `train/model_dispatch.py`
4. fixture / smoke / benchmark profile

絶対条件:
- `table_only` では使えない
- `grid_pack` adapter のみ
- model ごとに feature profile を増やさない
- compare / benchmark の mode-aware row を壊さない

作業完了時:
- 実装差分
- テスト結果
- U-NO / FFNO との差分要約

```

# 11_optional_geom_deeponet_siren_dual_mode_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Optional Phase: Geom-DeepONet-SIREN を dual-mode 基盤へ追加** します。

前提:
- 01〜08 が完了していること
- descriptor / latent lane の hook があること


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- `geom_deeponet_siren` を `table_plus_structure` 専用で追加する
- `coord_pack` + `descriptor_branch` の hybrid model として扱う
- `deeponet_plasma` mainline は壊さない

実装目標:
1. `src/plasma_surrogate/models/deeponet/geom_deeponet_siren.py`
2. `models/mlp/io.py`
3. `train/model_dispatch.py`
4. infer engine で hybrid input を受ける
5. fixture / smoke / benchmark profile

絶対条件:
- `deeponet_plasma` の strict mainline contract を崩さない
- new model は experimental family とする
- `table_only` では使えない
- feature profile と descriptor profile は registry から解決

作業完了時:
- 変更ファイル一覧
- hybrid adapter の説明
- 実行テスト

```

# 12_final_regression_docs_sync_prompt.txt

```text
あなたは、この repository を直接編集する実装担当です。今回は **Final Phase: regression / docs sync / cleanup** を行います。

前提:
- 必須フェーズ 01〜08 が終わっている
- optional 09〜11 は、実装していれば取り込む。未実装でもよい


最初に必ず、次のファイルをこの順で読んでください。
1. docs/11_dual_input_mode_runtime.md
2. docs/11_dual_input_mode_runtime/README.md
3. docs/11_dual_input_mode_runtime/CODEX_HANDOFF_START.txt
4. docs/11_dual_input_mode_runtime/00_user_view.md
5. docs/11_dual_input_mode_runtime/10_architecture_and_contract.md
6. docs/11_dual_input_mode_runtime/20_geometry_core_and_feature_lanes.md
7. docs/11_dual_input_mode_runtime/30_preprocess_train_infer_runtime.md
8. docs/11_dual_input_mode_runtime/40_model_policy_and_adapters.md
9. docs/11_dual_input_mode_runtime/50_geom_space_and_optimization.md
10. docs/11_dual_input_mode_runtime/60_yaml_templates_and_examples.md
11. docs/11_dual_input_mode_runtime/70_benchmark_tests_and_rollout.md
12. docs/11_dual_input_mode_runtime/80_migration_notes.md

次に少なくとも以下の既存コードを読むこと。
- src/plasma_surrogate/core/model_families.py
- src/plasma_surrogate/train/model_dispatch.py
- src/plasma_surrogate/preprocessing/runner.py
- src/plasma_surrogate/infer/engine.py
- src/plasma_surrogate/infer/optimize.py
- src/plasma_surrogate/data/geometry_provider.py
- src/plasma_surrogate/models/mlp/io.py
- src/plasma_surrogate/cli/workflows.py


今回のタスク:
- docs と実装の表記ズレを解消する
- YAML templates を最終形にそろえる
- tests / fixtures / benchmark examples を整理する
- dead code / duplicate constants / TODO の棚卸しをする

実装目標:
1. docs/11_dual_input_mode_runtime* と実コードの整合確認
2. `docs/README.md` への導線追記
3. `README_mainline.md` へ experimental note 追記
4. duplicate constants / duplicate validation の削減
5. regression test 追加

最終出力要件:
- 変更ファイル一覧
- docs sync summary
- dual-mode support matrix
- 実行した全テスト
- 未実装 optional models の一覧

```
