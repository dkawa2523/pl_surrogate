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
