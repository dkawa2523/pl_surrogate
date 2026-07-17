# GEC-CCP latest results and conference figures

このページは、現在の共通評価プロトコルによる学習結果、入力最適化結果、学会発表用図版への入口です。チェックポイントや前処理キャッシュではなく、第三者が確認できる図・表・評価記録だけを対象にしています。

## 1. 学習評価

- [8モデル・3 seed共通評価](../../runs/gec_ccp_nn_operator_comparison_v1/evaluation/index.md)
  - 78条件、Train / Validation / Test = 54 / 11 / 13
  - 8モデル×3 seedの24 run
  - Validation-onlyで代表seedを選び、13 Testケースの空間分布を評価
- [追加3モデルのsingle-run評価](../../runs/gec_ccp_nn_operator_comparison_v2_single_run_additions/evaluation/spatial_truth_pred_error/index.md)
  - CNO、UNet++、plasma-point DeepONetを同じn78評価契約でscreening
- [POD branch v2評価](../../runs/gec_ccp_pod_branch_tuned_v2/evaluation/spatial_truth_pred_error/index.md)
  - 採用したPOD coefficient branchの空間分布評価
- [モデル候補と評価契約](../gec_ccp_model_candidates/index.md)

本編用のモデル比較図は、[学会用独立パーツ集](independent_assets/index.md)の「モデル比較：GEC-CCP」から選択します。

## 2. 入力最適化

- [3計測ケース・3最適化手法](../../runs/gec_ccp_multi_sensor_input_optimization_cpu/index.md)
  - CMA-ES、TPE、Randomを各100回のCPU FNO評価で比較
  - 収束、真値誤差、パラメータ回復、計算コストを図示
- [5 optimizer seed再現性試験](../../runs/gec_ccp_multi_sensor_input_optimization_cpu_multiseed/index.md)
  - 同じ3計測に対して各手法15 run
  - 共通品質到達率、ERT、最終品質分布を評価

## 3. 学会発表用図版

- [最終採用図・評価条件・loss数式README](README.md)
- [採用図manifest](adopted_figures.csv)
- [第三者向け選択ガイド](independent_assets/index.md)
- [機械可読アセット一覧](independent_assets/asset_catalog.csv)
- [簡略図・計算式](index.md)

通常2D図は定量説明、平面斜視図は表紙・概要、SVGは編集、PDFは投稿、PNGはスライド貼付に使用します。

## 公開範囲

このブランチに含めるrun成果物はPNG、PDF、SVG、CSV、Markdown、JSONに限定しています。学習チェックポイント、NPZ、キャッシュ、標準出力ログは含めません。
