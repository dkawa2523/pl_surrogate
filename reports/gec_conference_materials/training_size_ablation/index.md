# GEC-CCP 学習データ数比較・学会用グラフ

[GEC-CCP学会用画像](../gec_ccp_index.md) → 学習データ数比較

固定した13件の補間テストデータを用い、学習条件数54・48・36・24に対するモデル精度を
比較した発表用セットです。全体指標2図と、同じMediumテストケースを使った空間誤差図を
同じフォルダに保存しています。

## 推奨する提示順

1. 全体の物理量値誤差
2. 全体の空間勾配誤差
3. FFNOとDenseMLPの電子密度空間誤差

## 全体指標

### Case-macro relative RMSE

[![Case-macro relative RMSE](ccp_case_macro_relative_rmse_by_training_size.png)](ccp_case_macro_relative_rmse_by_training_size.png)

[PNG](ccp_case_macro_relative_rmse_by_training_size.png) / [PDF](ccp_case_macro_relative_rmse_by_training_size.pdf) / [SVG](ccp_case_macro_relative_rmse_by_training_size.svg)

### Case-macro relative gradient error

[![Case-macro relative gradient error](ccp_case_macro_relative_gradient_error_by_training_size.png)](ccp_case_macro_relative_gradient_error_by_training_size.png)

[PNG](ccp_case_macro_relative_gradient_error_by_training_size.png) / [PDF](ccp_case_macro_relative_gradient_error_by_training_size.pdf) / [SVG](ccp_case_macro_relative_gradient_error_by_training_size.svg)

両図の縦軸は対数スケールです。Neural Network系を寒色・破線、Neural Operator系を
暖色・実線で表示しています。U-NOとDeepONet Plasmaは数値CSVには保持し、発表図からは
除外しています。

## 電子密度の空間誤差

[![FFNO and DenseMLP electron-density error](ccp_ffno_densemlp_ne_error_by_training_size.png)](ccp_ffno_densemlp_ne_error_by_training_size.png)

[PNG](ccp_ffno_densemlp_ne_error_by_training_size.png) / [PDF](ccp_ffno_densemlp_ne_error_by_training_size.pdf) / [SVG](ccp_ffno_densemlp_ne_error_by_training_size.svg) / [metadata](ccp_ffno_densemlp_ne_error_by_training_size_metadata.json)

共通ケースは `PP0=3, Td=0.03, gamma=0.04, PA=0.1` です。誤差は
`100 × (prediction - truth) / max(abs(truth))` とし、全パネルを共通の `±30%` で表示します。
赤は過大予測、青は過小予測です。

## 数値と出典

- [全モデル数値](case_macro_errors_by_training_size.csv)
- [全体図provenance](case_macro_plot_provenance.json)
- [空間誤差図metadata](ccp_ffno_densemlp_ne_error_by_training_size_metadata.json)
