

===== README.md =====

# Codex follow-up prompts for same-fidelity / low-data extensions

推奨順:
1. 02_ffno_prompt.txt
2. 03_unetpp_attn_prompt.txt
3. 04_coord_mlp_fourier_prompt.txt
4. 05_coord_mlp_siren_prompt.txt
5. 06_deeponet_pod_prompt.txt
6. 07_benchmark_compare_promotion_prompt.txt
7. 08_final_regression_docs_sync_prompt.txt

使い方:
- 1 回の Codex 実行では 1 ファイルだけを投げる。
- 各ステップの変更は commit するか stash してから次へ進む。
- 失敗時は直前のステップの prompt を再利用し、未完了部分だけを埋める。
- すべての prompt は、docs/09_same_fidelity_low_data_extensions* と既存 repo の契約を読む前提になっている。
- Prompt 01 は既に作成済みの unetpp 実装 prompt:
  /mnt/data/codex_first_prompt_same_fidelity_extensions.txt


===== 02_ffno_prompt.txt =====

あなたは、この repository を直接編集する実装担当です。目的は、既存契約を壊さずに「同一 fidelity・physics-informed なし」の少量データ向け拡張を段階実装することです。

この prompt は **Prompt 01 の unetpp 実装が完了している** 前提で使います。unetpp の変更が未 commit なら、まず保存してから進んでください。今回のタスクは **ffno だけ** です。unetpp_attn、coord_mlp、deeponet_pod には着手しないでください。

最初に必ず、次のファイルをこの順で読んでください。
1. docs/09_same_fidelity_low_data_extensions.md
2. docs/09_same_fidelity_low_data_extensions/README.md
3. docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md
4. docs/09_same_fidelity_low_data_extensions/20_ffno.md
5. docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md
6. 既存の関連実装として、少なくとも次も読むこと
   - src/plasma_surrogate/models/fno/simple_fno.py
   - src/plasma_surrogate/models/mlp/io.py
   - src/plasma_surrogate/train/model_dispatch.py
   - src/plasma_surrogate/cli/workflows.py
   - docs/04_training_models.md
   - docs/07_extension_guide.md
   - もし Prompt 01 で追加した unetpp 実装があるなら、その io / dispatch 変更も確認すること

今回のタスク:
- 新しい model_id `ffno` を追加する
- 既存 `fno` の外部契約を保つ
- `ffno` は mainline-candidate として扱う
- v1 は separable_1d の factorized spectral block のみでよい
- physics-informed 系の変更はしない

絶対条件:
- target 名を固定しない
- target ID の真実源は `dataset.targets[].id`
- target 順序の真実源は `output_layout.vars`
- target transform の真実源は `preprocessing.scalers.target_transforms.<var>`
- benchmark / compare の dynamic target 前提を壊さない
- `ne / ni / Te / phi` 固定コードを増やさない
- `coord_feature_pack` / `geom_feature_pack` の feature 順序は train / infer / benchmark で一致させる
- 既存 `fno`, `unet`, `unetpp`, `deeponet_plasma`, `global_mlp` の挙動を壊さない
- `tests/fixtures` はテストテンプレ扱い、運用 YAML 正本は `configs/experimental/same_fidelity_low_data/` に置く

実装内容の要件:
1. 新規ファイル `src/plasma_surrogate/models/fno/factorized_fno.py` を追加し、`FFNOBaseline` を実装する
   - `FNOBaseline` とできるだけ同じ wrapper API を持つこと
   - 必須メソッド:
     - `set_static_spatial_features`
     - `_feature_map`
     - `forward`
     - `predict_fields`
     - `backward_raw`
     - `state_dict_numpy`
     - `load_state_dict_numpy`
   - 最も安全なのは `simple_fno.py` の wrapper を踏襲し、内部ネットワークだけ factorized block に差し替えること
   - v1 は `spectral_h + spectral_w + pointwise_skip` の構成でよい
   - config は既存 `fno` の `n_modes`, `spectral_cfg.width`, `spectral_cfg.n_layers`, `dealias_ratio`, `taper_alpha`, `skip_filter` を最大限再利用する
   - 追加キーは `spectral_cfg.factorized_cfg.enabled`, `spectral_cfg.factorized_cfg.mode` の最小限でよい
   - `mode=separable_1d` だけ先に対応すればよい

2. `src/plasma_surrogate/models/mlp/io.py` を更新する
   - `build_model_from_name` に `ffno` を追加する
   - checkpoint save/load に `model_type == "ffno"` を追加する
   - meta は既存 `fno` と同型にし、少なくとも以下を保持すること
     - model_type
     - input_dim
     - grid_shape
     - out_channels
     - output_keys
     - input_feature_channels
     - with_rho_eff_head
     - n_modes
     - spectral_cfg
     - head_mlp
   - 既存 `fno` の checkpoint 形式との整合性を優先する

3. `src/plasma_surrogate/train/model_dispatch.py` を更新する
   - `ffno` を既存の unet-like torch branch / fno-like branch に自然に組み込む
   - `input_features.mode=geom_feature_pack` を mainline-candidate 条件として扱う
   - `require_pack` / distance transform / coord feature scaling は既存 `fno` と同じ helper / 流れを使う
   - strict validation は既存 `fno` と同一ルールを再利用する
   - 次を mainline-candidate の前提として検証すること
     - `train.ffno.target_family=allvars`
     - `train.ffno.target_vars == output_layout.vars`
     - `train.ffno.input_features.mode=geom_feature_pack`
     - `train.ffno.selection.mode=best_val_allvars_balance`

4. config テンプレートを追加・更新する
   - `configs/experimental/same_fidelity_low_data/ffno_template.yaml` を docs に合わせて使える状態にする
   - feature 既定は `[x, y, mask_plasma, distance_signed, distance_any]`
   - `input_features.mode=geom_feature_pack`
   - `require_pack=error`
   - `spectral_cfg.factorized_cfg.enabled: true`
   - `spectral_cfg.factorized_cfg.mode: separable_1d`

5. テストを追加する
   - unit test:
     - `tests/unit/models/test_ffno_model.py`
     - `tests/unit/train/test_model_dispatch_ffno.py`
   - integration smoke:
     - `tests/integration/test_train_ffno_smoke.py`
     - `tests/integration/test_infer_ffno_smoke.py`
   - 必要なら `tests/integration/test_cli_ffno_smoke.py`
   - build/load/train/infer が通ることを確認できるようにする

6. fixture を追加する
   - `tests/fixtures/benchmark_periodic_real_m7_ffno_isolated_mainline.yaml`
   - 既存 `m7_fno_isolated` と比較しやすい形に揃える
   - profile lock や dynamic target 契約を壊さないこと

7. benchmark / profile まわり
   - もし benchmark 側にモデル allowlist や isolated profile の追加が必要なら、最小差分で対応する
   - lock hash や dynamic metric の仕組みは変更しない
   - `ffno` を mainline-candidate として扱うが、既存 `fno` の結果列や compare ロジックは固定列化しない

実装方針:
- まず repo を読んで、既存 `fno` の wrapper API、checkpoint 形式、dispatch の流れを短く要約する
- その後で変更計画を箇条書きで示す
- その計画に沿ってコードを編集する
- 変更は最小差分を優先し、大規模 refactor はしない
- 既存 `fno` のコードを壊さず、内部ネットワークだけ差し替える方向を優先する
- 周辺一般化よりも、まず `ffno` の最小実装を green にする

作業完了時の出力要件:
- 変更したファイル一覧
- 追加した file 一覧
- 実装した strict / validation 内容
- 実行したテストと結果
- 未実装として残した点（あれば明記）

重要:
- 今回は **ffno だけ**
- unetpp_attn、coord_mlp、deeponet_pod はまだやらない
- physics-informed の変更、PINO、追加 loss、benchmark 固定列化は入れない
- 不明点があっても repo と docs を読んで、最も整合的な最小差分で前に進める


===== 03_unetpp_attn_prompt.txt =====

あなたは、この repository を直接編集する実装担当です。目的は、既存契約を壊さずに「同一 fidelity・physics-informed なし」の少量データ向け拡張を段階実装することです。

この prompt は **Prompt 01 の unetpp 実装が green** である前提です。今回のタスクは **unetpp_attn だけ** です。ffno、coord_mlp、deeponet_pod には着手しないでください。

最初に必ず、次のファイルをこの順で読んでください。
1. docs/09_same_fidelity_low_data_extensions.md
2. docs/09_same_fidelity_low_data_extensions/README.md
3. docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md
4. docs/09_same_fidelity_low_data_extensions/10_unetpp_and_attention.md
5. docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md
6. 既存と直前実装の関連ファイルとして、少なくとも次も読むこと
   - src/plasma_surrogate/models/unet/simple_unet.py
   - src/plasma_surrogate/models/unet/unetpp.py
   - src/plasma_surrogate/models/mlp/io.py
   - src/plasma_surrogate/train/model_dispatch.py
   - docs/04_training_models.md
   - docs/07_extension_guide.md
   - tests/unit/models/test_unetpp_model.py
   - tests/integration/test_train_unetpp_smoke.py
   - tests/integration/test_infer_unetpp_smoke.py

今回のタスク:
- 新しい model_id `unetpp_attn` を追加する
- `unetpp` の API と feature / target 契約をそのまま維持する
- `unetpp_attn` は mainline-candidate として扱う
- attention は skip tensor の gate のみでよい
- deep supervision は今回も実装しない。必要なら config 置き場だけ残して挙動は off のままでよい

絶対条件:
- target 名を固定しない
- target ID の真実源は `dataset.targets[].id`
- target 順序の真実源は `output_layout.vars`
- target transform の真実源は `preprocessing.scalers.target_transforms.<var>`
- benchmark / compare の dynamic target 前提を壊さない
- `ne / ni / Te / phi` 固定コードを増やさない
- `geom_feature_pack` の feature 順序は train / infer / benchmark で一致させる
- 既存 `unet`, `unetpp`, `fno`, `ffno`, `deeponet_plasma`, `global_mlp` を壊さない

実装内容の要件:
1. `src/plasma_surrogate/models/unet/unetpp.py` を更新する
   - `UNetPPBaseline` に attention を有効化できる構成を追加するか、最小差分の subclass / shared module で対応する
   - 新しい model_id `unetpp_attn` は API 上 `unetpp` と同じ wrapper を持つこと
   - 推奨は `conv_cfg.attention_cfg.enabled` を見て gate を入れる方式
   - gate は nested skip の concat 前に入れる
   - bottleneck attention や複雑な multi-head 化は不要
   - attention を無効にした場合、shape / API / checkpoint 互換性が `unetpp` と一致するようにする
   - 必須メソッドは `unetpp` と同じ
     - `set_static_spatial_features`
     - `_resolve_spatial_features`
     - `_feature_map`
     - `forward`
     - `predict_fields`
     - `backward_raw`
     - `state_dict_numpy`
     - `load_state_dict_numpy`

2. `src/plasma_surrogate/models/mlp/io.py` を更新する
   - `build_model_from_name` に `unetpp_attn` を追加する
   - `conv_cfg.attention_cfg.enabled = true` を builder 側で確実に反映させる
   - checkpoint save/load に `model_type == "unetpp_attn"` を追加する
   - meta は `unetpp` と同様に、少なくとも以下を保持すること
     - model_type
     - input_dim
     - grid_shape
     - out_channels
     - output_keys
     - backend
     - input_feature_channels
     - with_rho_eff_head
     - head_mlp
     - conv_cfg
     - output_heads

3. `src/plasma_surrogate/train/model_dispatch.py` を更新する
   - `unetpp_attn` を既存の unet-like torch branch に組み込む
   - strict validation は `unetpp` と同じ mainline-candidate ルールを再利用する
   - 次を mainline-candidate の前提として検証すること
     - `train.unetpp_attn.target_family=allvars`
     - `train.unetpp_attn.target_vars == output_layout.vars`
     - `train.unetpp_attn.input_features.mode=geom_feature_pack`
     - `train.unetpp_attn.model_cfg.output_heads.mode=shared`
     - `train.unetpp_attn.selection.mode=best_val_allvars_balance`
   - `require_pack` / distance transform / coord feature scaling は既存 `unetpp` と同じ helper / 流れを使う

4. config テンプレートを追加・更新する
   - `configs/experimental/same_fidelity_low_data/unetpp_attn_template.yaml` を docs に合わせて使える状態にする
   - feature 既定は `[x, y, mask_plasma, distance_signed, distance_any]`
   - `input_features.mode=geom_feature_pack`
   - `require_pack=error`
   - `model_cfg.conv_cfg.attention_cfg.enabled: true`
   - reduction や gate activation は docs の最小構成に合わせる

5. テストを追加する
   - unit test:
     - `tests/unit/models/test_unetpp_attn_model.py` を追加してよいし、`test_unetpp_model.py` に拡張してもよい
     - `tests/unit/train/test_model_dispatch_unetpp_attn.py`
   - integration smoke:
     - `tests/integration/test_train_unetpp_attn_smoke.py`
     - `tests/integration/test_infer_unetpp_attn_smoke.py`
   - 注意:
     - attention 有効時の forward shape
     - checkpoint roundtrip
     - strict contract validation
     を最低限確認すること

6. fixture を追加する
   - `tests/fixtures/benchmark_periodic_real_m7_unetpp_attn_isolated_mainline.yaml`
   - profile lock や dynamic target 契約を壊さないこと
   - 既存 `unetpp` fixture と比較しやすい構造に揃える

実装方針:
- まず repo を読んで、現在の `unetpp` 実装の API と builder / dispatch 連携を短く要約する
- その後で変更計画を箇条書きで示す
- その計画に沿ってコードを編集する
- 変更は最小差分を優先し、大規模 refactor はしない
- attention のためだけに trainer や infer engine を大改造しない
- deep supervision は今回入れない

作業完了時の出力要件:
- 変更したファイル一覧
- 追加した file 一覧
- 実装した strict / validation 内容
- 実行したテストと結果
- 未実装として残した点（あれば明記）

重要:
- 今回は **unetpp_attn だけ**
- ffno、coord_mlp、deeponet_pod はまだやらない
- physics-informed の変更や benchmark 固定列化は入れない


===== 04_coord_mlp_fourier_prompt.txt =====

あなたは、この repository を直接編集する実装担当です。目的は、既存契約を壊さずに「同一 fidelity・physics-informed なし」の少量データ向け拡張を段階実装することです。

この prompt は、`unetpp` / `ffno` 系の変更が保存済みである前提で使います。今回のタスクは **coord_mlp_fourier だけ** です。SIREN 版や deeponet_pod には着手しないでください。

最初に必ず、次のファイルをこの順で読んでください。
1. docs/09_same_fidelity_low_data_extensions.md
2. docs/09_same_fidelity_low_data_extensions/README.md
3. docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md
4. docs/09_same_fidelity_low_data_extensions/30_coord_mlp_family.md
5. docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md
6. 既存の関連実装として、少なくとも次も読むこと
   - src/plasma_surrogate/models/mlp/global_mlp.py
   - src/plasma_surrogate/models/mlp/io.py
   - src/plasma_surrogate/train/model_dispatch.py
   - src/plasma_surrogate/models/unet/simple_unet.py
   - src/plasma_surrogate/cli/workflows.py
   - docs/04_training_models.md
   - docs/07_extension_guide.md

今回のタスク:
- 新しい model_id `coord_mlp_fourier` を追加する
- これは **experimental** として扱う
- NumPy `global_mlp` branch には入れず、torch-style full-field wrapper として実装する
- `coord_feature_pack` / `geom_feature_pack` を使う geometry-aware な座標デコーダにする
- v1 は full-grid decode のみでよい
- point sampling / chunked decode / advanced caching はまだ実装しない

絶対条件:
- target 名を固定しない
- target ID の真実源は `dataset.targets[].id`
- target 順序の真実源は `output_layout.vars`
- target transform の真実源は `preprocessing.scalers.target_transforms.<var>`
- benchmark / compare の dynamic target 前提を壊さない
- `ne / ni / Te / phi` 固定コードを増やさない
- `global_mlp` の既存 NumPy branch を壊さない
- `geom_feature_pack` の feature 順序は train / infer / benchmark で一致させる
- experimental だが、feature pack が無いときの fallback で曖昧な挙動を入れない

実装内容の要件:
1. 新規ファイル `src/plasma_surrogate/models/mlp/coord_mlp_torch.py` を追加し、`CoordMLPTorch` を実装する
   - 必須メソッド:
     - `set_static_spatial_features`
     - `_resolve_spatial_features`
     - `_build_decoder_input`
     - `forward`
     - `predict_fields`
     - `backward_raw`
     - `state_dict_numpy`
     - `load_state_dict_numpy`
   - 構造は次でよい
     - cond vector -> cond encoder -> z_cond
     - coord feature pack -> Fourier embedding -> z_point
     - concat(z_cond, z_point) -> point decoder -> field values
     - reshape -> `[B, C_out, H, W]`
   - v1 は full-grid decode のみ
   - pointwise decoder は shared multi-target head でよい
   - deterministic Fourier embedding でよい
   - config 例:
     - `embedding.type = fourier`
     - `embedding.n_frequencies = 8`
     - `embedding.include_raw = true`
     - `embedding.frequency_scale = 10.0`

2. `src/plasma_surrogate/models/mlp/io.py` を更新する
   - `build_model_from_name` に `coord_mlp_fourier` を追加する
   - checkpoint save/load に `model_type == "coord_mlp_fourier"` を追加する
   - meta には少なくとも以下を保持すること
     - model_type
     - input_dim
     - grid_shape
     - out_channels
     - output_keys
     - input_feature_channels
     - model_cfg
   - ここではまだ `coord_mlp_siren` を実装しないが、将来追加しやすい API にしてよい

3. `src/plasma_surrogate/train/model_dispatch.py` を更新する
   - `coord_mlp_fourier` を unet-like torch branch に組み込む
   - experimental validator として次を明示的に検証すること
     - `target_family == allvars`
     - `target_vars == output_layout.vars`
     - `input_features.mode == geom_feature_pack`
     - `require_pack == error` を推奨し、少なくとも pack 不在を silent fallback しない
     - feature list に重複がない
   - 既存 `unet` / `fno` と同じ feature pack 解決 helper を再利用できるなら再利用する
   - `legacy_xy` fallback は許さない
   - 既定 feature は `[x, y, mask_plasma, distance_signed, distance_any]`

4. config テンプレートを追加・更新する
   - `configs/experimental/same_fidelity_low_data/coord_mlp_fourier_template.yaml` を docs に合わせて使える状態にする
   - `input_features.mode=geom_feature_pack`
   - `require_pack=error`
   - feature 既定は `[x, y, mask_plasma, distance_signed, distance_any]`
   - `model_cfg.embedding.type = fourier`

5. テストを追加する
   - unit test:
     - `tests/unit/models/test_coord_mlp_model.py`
       - Fourier build
       - static feature shape validation
       - forward shape
       - checkpoint roundtrip
     - `tests/unit/train/test_model_dispatch_coord_mlp.py`
       - `geom_feature_pack` が無いと落ちる
       - feature duplicate validation
   - integration smoke:
     - `tests/integration/test_train_coord_mlp_smoke.py`
     - `tests/integration/test_infer_coord_mlp_smoke.py`
   - v1 では Fourier 版のみ green にする

6. fixture を追加する
   - `tests/fixtures/benchmark_periodic_real_m7_coord_mlp_fourier_experimental.yaml`
   - これは experimental fixture として扱う
   - mainline profile lock には入れない

実装方針:
- まず repo を読んで、`global_mlp` の弱点と、torch-style wrapper に入れる理由を短く要約する
- その後で変更計画を箇条書きで示す
- その計画に沿ってコードを編集する
- 変更は最小差分を優先し、大規模 refactor はしない
- まず Fourier 版だけを完成させ、その後の SIREN 追加をしやすい形にとどめる
- global_mlp の NumPy 実装を巻き込んだ一般化はしない

作業完了時の出力要件:
- 変更したファイル一覧
- 追加した file 一覧
- 実装した validation 内容
- 実行したテストと結果
- 次の prompt で SIREN を入れるために残した拡張点（あれば明記）

重要:
- 今回は **coord_mlp_fourier だけ**
- `coord_mlp_siren` はまだやらない
- deeponet_pod にはまだ触らない
- physics-informed の変更や benchmark 固定列化は入れない


===== 05_coord_mlp_siren_prompt.txt =====

あなたは、この repository を直接編集する実装担当です。目的は、既存契約を壊さずに「同一 fidelity・physics-informed なし」の少量データ向け拡張を段階実装することです。

この prompt は **coord_mlp_fourier の実装が green** である前提です。今回のタスクは **coord_mlp_siren だけ** です。Fourier 版や他モデルの大改造はしないでください。

最初に必ず、次のファイルをこの順で読んでください。
1. docs/09_same_fidelity_low_data_extensions.md
2. docs/09_same_fidelity_low_data_extensions/README.md
3. docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md
4. docs/09_same_fidelity_low_data_extensions/30_coord_mlp_family.md
5. docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md
6. 直前実装と既存関連として、少なくとも次も読むこと
   - src/plasma_surrogate/models/mlp/coord_mlp_torch.py
   - src/plasma_surrogate/models/mlp/io.py
   - src/plasma_surrogate/train/model_dispatch.py
   - tests/unit/models/test_coord_mlp_model.py
   - tests/integration/test_train_coord_mlp_smoke.py
   - docs/04_training_models.md
   - docs/07_extension_guide.md

今回のタスク:
- 新しい model_id `coord_mlp_siren` を追加する
- これは **experimental** として扱う
- `coord_mlp_fourier` と同じ wrapper API を維持する
- 相違点は主に embedding / activation / initialization のみとする
- v1 は full-grid decode のみでよい
- point sampling / chunked decode / mixed embedding はまだ実装しない

絶対条件:
- target 名を固定しない
- target ID の真実源は `dataset.targets[].id`
- target 順序の真実源は `output_layout.vars`
- target transform の真実源は `preprocessing.scalers.target_transforms.<var>`
- benchmark / compare の dynamic target 前提を壊さない
- `ne / ni / Te / phi` 固定コードを増やさない
- `coord_mlp_fourier` の既存動作を壊さない
- `geom_feature_pack` の feature 順序は train / infer / benchmark で一致させる

実装内容の要件:
1. `src/plasma_surrogate/models/mlp/coord_mlp_torch.py` を更新する
   - `coord_mlp_siren` を同じ family の設定違いとして実装する
   - SIREN 用の活性と初期化を追加する
   - 必須メソッドは Fourier 版と同じまま維持する
     - `set_static_spatial_features`
     - `_resolve_spatial_features`
     - `_build_decoder_input`
     - `forward`
     - `predict_fields`
     - `backward_raw`
     - `state_dict_numpy`
     - `load_state_dict_numpy`
   - config 例:
     - `embedding.type = none`
     - `siren.enabled = true`
     - `siren.w0_initial = 30.0`
     - `siren.w0_hidden = 1.0`
   - 実装上の注意:
     - first layer は `w0_initial`
     - hidden layer は `w0_hidden`
     - SIREN 用初期化を実装する
     - wrapper API は Fourier 版と同一に保つ

2. `src/plasma_surrogate/models/mlp/io.py` を更新する
   - `build_model_from_name` に `coord_mlp_siren` を追加する
   - `coord_mlp_fourier` と同じ builder を使いながら、SIREN 用 config を有効化する
   - checkpoint save/load に `model_type == "coord_mlp_siren"` を追加する
   - meta には少なくとも以下を保持すること
     - model_type
     - input_dim
     - grid_shape
     - out_channels
     - output_keys
     - input_feature_channels
     - model_cfg

3. `src/plasma_surrogate/train/model_dispatch.py` を更新する
   - `coord_mlp_siren` を `coord_mlp_fourier` と同じ experimental branch に追加する
   - validator は Fourier 版と同じでよい
   - `geom_feature_pack` 必須、`legacy_xy` fallback 不可、feature duplicate 不可 を守る
   - `target_family == allvars`
   - `target_vars == output_layout.vars`
   を維持する

4. config テンプレートを追加・更新する
   - `configs/experimental/same_fidelity_low_data/coord_mlp_siren_template.yaml` を docs に合わせて使える状態にする
   - `input_features.mode=geom_feature_pack`
   - `require_pack=error`
   - feature 既定は `[x, y, mask_plasma, distance_signed, distance_any]`
   - `model_cfg.siren.enabled = true`

5. テストを追加・更新する
   - unit test:
     - `tests/unit/models/test_coord_mlp_model.py`
       - SIREN build
       - SIREN forward shape
       - checkpoint roundtrip
     - `tests/unit/train/test_model_dispatch_coord_mlp.py`
       - `coord_mlp_siren` の validator も通る / 落ちるケースを追加
   - integration smoke:
     - 既存 `test_train_coord_mlp_smoke.py` / `test_infer_coord_mlp_smoke.py` を parametrize するか
     - もしくは `tests/integration/test_train_coord_mlp_siren_smoke.py`
     - `tests/integration/test_infer_coord_mlp_siren_smoke.py`
   - `coord_mlp_fourier` の既存 green を壊さないこと

6. fixture を追加する
   - `tests/fixtures/benchmark_periodic_real_m7_coord_mlp_siren_experimental.yaml`
   - experimental fixture として扱う
   - mainline profile lock には入れない

実装方針:
- まず repo と現行 `coord_mlp_fourier` 実装を読んで、流用できる部分を短く要約する
- その後で変更計画を箇条書きで示す
- その計画に沿ってコードを編集する
- 変更は最小差分を優先し、大規模 refactor はしない
- Fourier 版の共通 wrapper を維持し、SIREN 版は config 差し替え中心で入れる

作業完了時の出力要件:
- 変更したファイル一覧
- 追加した file 一覧
- 実装した validation 内容
- 実行したテストと結果
- Fourier 版と SIREN 版の差分を簡潔に明記

重要:
- 今回は **coord_mlp_siren だけ**
- deeponet_pod にはまだ触らない
- physics-informed の変更や benchmark 固定列化は入れない


===== 06_deeponet_pod_prompt.txt =====

あなたは、この repository を直接編集する実装担当です。目的は、既存契約を壊さずに「同一 fidelity・physics-informed なし」の少量データ向け拡張を段階実装することです。

この prompt は、前段のモデル追加が保存済みである前提で使います。今回のタスクは **deeponet_pod だけ** です。既存 `deeponet_plasma` の strict mainline path は壊さないでください。

最初に必ず、次のファイルをこの順で読んでください。
1. docs/09_same_fidelity_low_data_extensions.md
2. docs/09_same_fidelity_low_data_extensions/README.md
3. docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md
4. docs/09_same_fidelity_low_data_extensions/40_pod_deeponet.md
5. docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md
6. 既存の関連実装として、少なくとも次も読むこと
   - src/plasma_surrogate/models/deeponet/plasma_operator_torch.py
   - src/plasma_surrogate/models/deeponet/poisson_head_torch.py
   - src/plasma_surrogate/models/mlp/io.py
   - src/plasma_surrogate/train/model_dispatch.py
   - src/plasma_surrogate/train/torch_trainer.py
   - docs/04_training_models.md
   - docs/07_extension_guide.md

今回のタスク:
- 新しい model_id `deeponet_pod` を追加する
- これは **experimental** として扱う
- reduced-order operator として、full-field の代わりに POD basis 上の係数を学ぶ
- v1 は **train split 上で basis を fitし、その basis を checkpoint に埋め込む**
- infer 時に新しい preprocess artifact は要求しない
- existing `deeponet_plasma` の plain strict mainline contract には無理に入れない

絶対条件:
- target 名を固定しない
- target ID の真実源は `dataset.targets[].id`
- target 順序の真実源は `output_layout.vars`
- target transform の真実源は `preprocessing.scalers.target_transforms.<var>`
- benchmark / compare の dynamic target 前提を壊さない
- `ne / ni / Te / phi` 固定コードを増やさない
- basis fitting に validation / test split を混ぜない
- checkpoint だけで infer できるようにする
- existing `deeponet_plasma` branch の strict validation や physics 拡張は触らない

実装内容の要件:
1. 新規ファイル `src/plasma_surrogate/models/deeponet/pod_deeponet_torch.py` を追加し、`PODDeepONetTorch` を実装する
   - 入力:
     - branch: 条件ベクトル `cond`
     - trunk: 固定された POD basis
   - 出力:
     - 各 target ごとの POD coefficient
     - basis 再構成により `[B, C_out, H, W]` を返す
   - 必須メソッド:
     - `forward`
     - `predict_fields`
     - `backward_raw`
     - `state_dict_numpy`
     - `load_state_dict_numpy`
     - `to_meta`（あれば望ましい）
   - v1 は target ごとに独立 basis を持つ
   - basis は `basis[var]`, `mean[var]`, `rank[var]` を保持する
   - 既定 rank は 32 だが、sample 数と `H*W` に応じて clamp する

2. basis fitting helper を実装する
   - 置き場は `pod_deeponet_torch.py` 内 helper でも、最小限の別 util でもよい
   - fit 対象は **`ctx.y_scaled[ctx.tr]` だけ**
   - target ごとに
     - snapshots = `[N, H*W]`
     - 必要なら center
     - SVD
     - top-r を basis として保持
   - 保存するもの:
     - `basis[var]`: `[rank, H, W]`
     - `mean[var]`: `[H, W]`
     - `rank[var]`

3. `src/plasma_surrogate/train/model_dispatch.py` を更新する
   - `deeponet_pod` 用の専用分岐を追加する
   - builder に渡す前に `ctx.y_scaled[ctx.tr]` から basis を fit する
   - `model_cfg["__pod_basis__"]`, `model_cfg["__pod_mean__"]` のような内部受け渡しでよい
   - experimental validator として次を検証すること
     - `target_family == allvars`
     - `target_vars == output_layout.vars`
     - `model_cfg.basis.rank >= 1`
     - `model_cfg.basis.fit_scope == train_only`
   - 既存 `deeponet_plasma` の strict mainline 分岐は壊さない

4. `src/plasma_surrogate/models/mlp/io.py` を更新する
   - `build_model_from_name` に `deeponet_pod` を追加する
   - builder は `__pod_basis__`, `__pod_mean__` を受け取って `PODDeepONetTorch` を構築する
   - checkpoint save/load に `model_type == "deeponet_pod"` を追加する
   - basis と mean は checkpoint に埋め込む
   - meta には少なくとも以下を保持すること
     - model_type
     - input_dim
     - grid_shape
     - out_channels
     - output_keys
     - model_cfg
     - basis_keys
     - basis_rank_by_var

5. config テンプレートを追加・更新する
   - `configs/experimental/same_fidelity_low_data/deeponet_pod_template.yaml` を docs に合わせて使える状態にする
   - basis config の最低限:
     - `rank: 32`
     - `fit_scope: train_only`
     - `per_var: true`
     - `center: true`
   - `target_family: allvars`
   - `target_vars == output_layout.vars`

6. テストを追加する
   - unit test:
     - `tests/unit/models/test_pod_deeponet_model.py`
       - basis fit helper
       - forward / reconstruction shape
       - checkpoint roundtrip
     - `tests/unit/train/test_model_dispatch_pod_deeponet.py`
       - train split only で basis fit される
       - invalid rank / invalid fit_scope を落とす
   - integration smoke:
     - `tests/integration/test_train_pod_deeponet_smoke.py`
     - `tests/integration/test_infer_pod_deeponet_smoke.py`
   - checkpoint だけで infer が通ることを確認する

7. fixture を追加する
   - `tests/fixtures/benchmark_periodic_real_m7_deeponet_pod_experimental.yaml`
   - experimental fixture として扱う
   - mainline profile lock には入れない

実装方針:
- まず repo を読んで、既存 `deeponet_plasma` の外部契約と strict mainline 条件を短く要約する
- その後で、なぜ `deeponet_pod` を専用分岐にするかを一言で明確にする
- 変更計画を箇条書きで示し、その計画に沿ってコードを編集する
- 変更は最小差分を優先し、大規模 refactor はしない
- infer 用に新しい preprocess artifact を作らず、checkpoint 埋め込みで閉じることを優先する

作業完了時の出力要件:
- 変更したファイル一覧
- 追加した file 一覧
- 実装した validation 内容
- basis fit の流れ（どの split を使ったか）
- 実行したテストと結果
- 未実装として残した点（あれば明記）

重要:
- 今回は **deeponet_pod だけ**
- 既存 `deeponet_plasma` の physics / boundary / operator strict path は触らない
- physics-informed の変更や benchmark 固定列化は入れない


===== 07_benchmark_compare_promotion_prompt.txt =====

あなたは、この repository を直接編集する実装担当です。目的は、ここまで追加した same-fidelity / low-data 拡張モデルを、repo の benchmark / compare / config 動線に **最小差分で** つなぐことです。

この prompt は、少なくとも次のモデル実装が保存済みである前提で使います。
- unetpp
- ffno
- unetpp_attn
- coord_mlp_fourier
- coord_mlp_siren
- deeponet_pod

今回のタスクは **benchmark / compare / config 整備だけ** です。新しい model class の追加や physics-informed 系変更はしないでください。

最初に必ず、次のファイルをこの順で読んでください。
1. docs/09_same_fidelity_low_data_extensions.md
2. docs/09_same_fidelity_low_data_extensions/README.md
3. docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md
4. docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md
5. configs/experimental/same_fidelity_low_data/README.md
6. configs/experimental/same_fidelity_low_data/benchmark_compare_template.yaml
7. 既存の関連実装として、少なくとも次も読むこと
   - src/plasma_surrogate/benchmark/runner.py
   - src/plasma_surrogate/benchmark/profiles.py
   - scripts/compare_selected_models.py
   - docs/04_training_models.md
   - docs/07_extension_guide.md
   - ここまでに追加した tests/fixtures と configs/experimental/same_fidelity_low_data/*.yaml

今回のタスク:
- ここまでに追加したモデルの **isolated benchmark fixture** と **運用向け config テンプレート** を整える
- dynamic target 前提を壊さずに compare 動線を確認する
- mainline-candidate は `unetpp`, `unetpp_attn`, `ffno` のみ
- experimental は `coord_mlp_fourier`, `coord_mlp_siren`, `deeponet_pod`
- experimental を mainline profile lock へ昇格させない
- 追加評価指標を入れる場合も leaderboard 必須列は変えない

絶対条件:
- benchmark の profile lock / resolved config / lock hash の仕組みを壊さない
- compare の dynamic key 保持を壊さない
- 固定列 `test_rmse_phi` のみを前提にした比較を入れない
- `output_layout.vars` と別順の target を許容しない
- `global_reference_mode=frozen` を使う場合は `global_mlp` 行前提を守る
- 既存 mainline fixture の意味を変えない

実装内容の要件:
1. fixture / config の棚卸しと整合性修正
   - 既に追加した fixture / template のパスと model_id が一致しているか確認する
   - 足りないものがあれば追加する
   - テスト用 fixture は `tests/fixtures`
   - 運用向け正本テンプレは `configs/experimental/same_fidelity_low_data/`
   - 少なくとも次の名前を揃えること
     - `benchmark_periodic_real_m7_unetpp_isolated_mainline.yaml`
     - `benchmark_periodic_real_m7_unetpp_attn_isolated_mainline.yaml`
     - `benchmark_periodic_real_m7_ffno_isolated_mainline.yaml`
     - `benchmark_periodic_real_m7_coord_mlp_fourier_experimental.yaml`
     - `benchmark_periodic_real_m7_coord_mlp_siren_experimental.yaml`
     - `benchmark_periodic_real_m7_deeponet_pod_experimental.yaml`

2. benchmark / profile 側の最小対応
   - もし `benchmark/profiles.py` や runner 側に model allowlist / isolated profile の追加が必要なら、最小差分で対応する
   - mainline-candidate の `unetpp`, `unetpp_attn`, `ffno` のみを mainline 文脈へ接続してよい
   - experimental モデルは isolated experimental fixture で回すだけにとどめる
   - lock semantics は変えない
   - dynamic target 集計ロジックは変えない

3. compare 向け example config の整備
   - `configs/experimental/same_fidelity_low_data/benchmark_compare_template.yaml` を実装済み model_id と整合するよう確認する
   - 必要なら compare 例を追加してよい
   - ただし compare は dynamic key 前提のままにする
   - `objective_metric=auto_primary` を壊さない
   - `global_reference_mode=frozen` を使う場合の前提をコメントで明示してよい

4. optional: 補助評価 CSV の接続
   - 追加する場合は leaderboard 必須列にはしない
   - 許容される補助指標:
     - boundary band RMSE
     - plasma mask 内 RMSE
     - plasma mask 外 RMSE
     - POD reconstruction error（deeponet_pod のみ）
     - high-frequency band error（ffno / coord_mlp 系比較用）
   - 既存 benchmark CSV スキーマを壊すなら今回はやらない

5. テストを追加・更新する
   - 必要なら benchmark / compare の軽量 smoke を追加する
   - ただし重すぎる統合テストは避ける
   - 推奨:
     - isolated fixture が parse できる
     - runner / compare が dynamic target を壊さない
     を確認する軽量テスト
   - 既存の model-level smoke を再利用できるなら新規テストは最小限でよい

実装方針:
- まず repo を読んで、benchmark / compare が固定列ではなく dynamic target 前提である点を短く要約する
- その後で fixture / config / optional code 修正の計画を箇条書きで示す
- その計画に沿ってコードと YAML を編集する
- 変更は最小差分を優先し、大規模 refactor はしない
- benchmark 本体に大きな仕様変更が必要なら、今回はやらず TODO として残す

作業完了時の出力要件:
- 変更したファイル一覧
- 追加・更新した fixture / config 一覧
- benchmark / compare に加えた最小修正の要約
- 実行したテストと結果
- optional 評価指標を見送った場合は、その理由を明記

重要:
- 今回は **benchmark / compare / config 整備だけ**
- 新 model class の追加や physics-informed の変更はしない
- dynamic target 契約と profile lock を壊さない


===== 08_final_regression_docs_sync_prompt.txt =====

あなたは、この repository を直接編集する実装担当です。目的は、ここまで追加した same-fidelity / low-data 拡張の **最終整合確認** を行い、docs / tests / config / registry のズレを潰して、VS Code 上の別開発者がそのまま読める状態に仕上げることです。

この prompt は、前段の model 実装と benchmark / compare 整備が保存済みである前提で使います。今回のタスクは **最終 regression / docs sync / cleanup** だけです。新モデルや新アルゴリズムの追加はしないでください。

最初に必ず、次のファイルをこの順で読んでください。
1. docs/09_same_fidelity_low_data_extensions.md
2. docs/09_same_fidelity_low_data_extensions/README.md
3. docs/09_same_fidelity_low_data_extensions/00_common_guardrails.md
4. docs/09_same_fidelity_low_data_extensions/10_unetpp_and_attention.md
5. docs/09_same_fidelity_low_data_extensions/20_ffno.md
6. docs/09_same_fidelity_low_data_extensions/30_coord_mlp_family.md
7. docs/09_same_fidelity_low_data_extensions/40_pod_deeponet.md
8. docs/09_same_fidelity_low_data_extensions/90_benchmark_and_tests.md
9. docs/04_training_models.md
10. docs/07_extension_guide.md
11. ここまでに変更した models / io / dispatch / benchmark / tests / configs 一式

今回のタスク:
- 実装済み model_id / YAML キー / docs 記述 / checkpoint meta / fixture 名のズレを洗い出して修正する
- 追加モデルと既存 mainline の回帰テストをまとめて流し、壊していないことを確認する
- docs の導線を整え、Codex や別開発者が repo 内ドキュメントだけで追える状態にする
- 未実装 TODO を明示的に残す

対象モデル:
- unetpp
- unetpp_attn
- ffno
- coord_mlp_fourier
- coord_mlp_siren
- deeponet_pod

絶対条件:
- 既存 `global_mlp`, `unet`, `fno`, `deeponet_plasma` の挙動を壊さない
- target 名固定コードを増やさない
- `dataset.targets[].id`, `output_layout.vars`, `preprocessing.scalers.target_transforms.<var>` を真実源とする前提を壊さない
- benchmark / compare の dynamic target 前提を壊さない
- docs と実装の齟齬を残したまま「完了」としない

実装内容の要件:
1. docs sync
   - 必要なら `docs/04_training_models.md` に追加モデルの位置づけを追記する
   - 必要なら `docs/07_extension_guide.md` に builder / dispatch / tests / fixture の接続点を追記する
   - `docs/09_same_fidelity_low_data_extensions.md` から各個別仕様への導線を確認する
   - docs 内の path, model_id, YAML key, fixture 名が実装と一致しているか確認する

2. config / fixture / registry sync
   - `configs/experimental/same_fidelity_low_data/*.yaml` の model 名や key が実装と一致しているか確認する
   - `tests/fixtures/*.yaml` の model_id が実装と一致しているか確認する
   - `models/mlp/io.py` の builder / load / save registry と docs が一致しているか確認する
   - `train/model_dispatch.py` の分岐名と docs が一致しているか確認する

3. regression tests
   - 新規追加モデルの targeted tests を流す
   - 加えて、影響を受ける既存モデルの軽量回帰も流す
   - 目安:
     - unit: unet / fno / global_mlp / deeponet_plasma に関係するもの
     - integration smoke: 新規モデル一式 + 既存 `unet`, `fno` の少なくとも一部
   - テストが重すぎる場合は、最小限の回帰セットを選び、その理由を明記する

4. TODO / follow-up の明示
   - 今回まだ入れていないものを `TODO` として docs に残してよい
   - 例:
     - unetpp deep supervision
     - coord_mlp point sampling / chunked decode
     - deeponet_pod の preprocess artifact 化
     - mainline 昇格条件の再評価
   - TODO は「今後の拡張」と「今回 intentionally omitted」を区別して書く

実装方針:
- まず repo を読んで、今回追加したモデル群の registry と docs が一致しているか観点を列挙する
- その後で変更計画を箇条書きで示す
- その計画に沿って docs / tests / config / registry を調整する
- 変更は最小差分を優先し、大規模リファクタはしない
- 既存テストが不安定なら、それを回避せず原因と範囲を明示する

作業完了時の出力要件:
- 変更したファイル一覧
- docs / config / fixture / registry の同期内容の要約
- 実行したテストと結果
- 残した TODO と理由
- 既知の制約や今後の follow-up を簡潔に列挙

重要:
- 今回は **最終 regression / docs sync / cleanup だけ**
- 新しいモデルや physics-informed 機能は追加しない
- docs と実装の一貫性を最優先にする
