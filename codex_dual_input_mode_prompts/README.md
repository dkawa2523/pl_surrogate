# Codex Prompts for Dual Input Mode Runtime

このディレクトリは、`table_only` と `table_plus_structure` を明確に分ける拡張を、Codex に段階実装させるための prompt 一式です。

## 実行順

1. `prompts/01_phase0_input_mode_contract_prompt.txt`
2. `prompts/02_phase1_table_only_runtime_prompt.txt`
3. `prompts/03_phase1_structure_registry_prompt.txt`
4. `prompts/04_phase2_preprocess_train_infer_prompt.txt`
5. `prompts/05_phase2_model_policy_dispatch_prompt.txt`
6. `prompts/06_phase3_benchmark_compare_prompt.txt`
7. `prompts/07_phase4_provider_geom_space_prompt.txt`
8. `prompts/08_phase5_descriptor_latent_prompt.txt`
9. `prompts/09_optional_u_no_dual_mode_prompt.txt`
10. `prompts/10_optional_cno_dual_mode_prompt.txt`
11. `prompts/11_optional_geom_deeponet_siren_dual_mode_prompt.txt`
12. `prompts/12_final_regression_docs_sync_prompt.txt`

## 実行ルール

- 1 回の Codex 実行で 1 本だけ使う
- 各段階で commit または stash を取ってから次へ進む
- `table_only` と `table_plus_structure` の mode 分離を壊す大規模 refactor は避ける
- truth source を壊さない
- target 名の固定化をしない

## 安全な停止点

- 06 まで  
  dual-mode runtime + benchmark separation まで完成

- 08 まで  
  descriptor lane / reduced-order dual-mode まで完成

- 12 まで  
  optional future models を含めた統合作業まで完了
