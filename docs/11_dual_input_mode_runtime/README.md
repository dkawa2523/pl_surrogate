# Dual Input Mode Runtime Spec

このディレクトリは、**ユーザー視点の mode 選択**と、**アーキテクト視点の runtime 分離**を同時に扱う仕様書です。

## Codex の推奨読順

1. `CODEX_HANDOFF_START.txt`
2. `00_user_view.md`
3. `10_architecture_and_contract.md`
4. `20_geometry_core_and_feature_lanes.md`
5. `30_preprocess_train_infer_runtime.md`
6. `40_model_policy_and_adapters.md`
7. `50_geom_space_and_optimization.md`
8. `60_yaml_templates_and_examples.md`
9. `70_benchmark_tests_and_rollout.md`
10. `80_migration_notes.md`

## この仕様の主眼

この章は「構造特徴量化の良し悪し」を議論する前に、まず **runtime の入口を 2 mode に固定する** ことを優先します。

- `table_only`
  - 入力の意味論は cond-only
  - geometry は output context
  - optimize は `space` のみ

- `table_plus_structure`
  - cond + structure
  - feature profile / descriptor profile / latent profile を使う
  - optimize は `space` に加えて `geom_space` を使える

## 実装フェーズ

### Phase 0
`runtime.input_mode` 契約と共通 registry を追加する

### Phase 1
`table_only` を first-class にする  
geometry を output context としてのみ使う path を明確化する

### Phase 2
`table_plus_structure` の feature profile / model adapter / metadata を実装する

### Phase 3
benchmark / compare / rollout を input_mode aware にする

### Phase 4
`geom_space` と provider runtime を structure mode 専用で実装する

### Phase 5
descriptor / latent lane を `deeponet_pod` などへ足す

## 安全な停止点

- Phase 1 完了時点  
  `table_only` と `table_plus_structure` のルーティングが分かれ、既存モデルが壊れていない

- Phase 3 完了時点  
  benchmark / compare / checkpoint metadata が mode aware になっている

- Phase 5 完了時点  
  descriptor / latent lane が使えるが、future models 追加はまだ行っていない
