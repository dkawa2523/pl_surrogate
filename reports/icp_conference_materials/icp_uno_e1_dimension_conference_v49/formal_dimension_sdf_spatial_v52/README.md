# Formal Dimension／Formal SDF 空間分布比較

最新の正式モデル（seed 1237、200 epochs、同じ957ケース・固定split）について、未知構造3例のCOMSOL真値、予測値、符号付き誤差を同一色軸で比較します。

## 推奨図

[DimensionとSDFの直接比較（PNG）](22_formal_dimension_sdf_truth_prediction_error.png)／[SVG](22_formal_dimension_sdf_truth_prediction_error.svg)／[PDF](22_formal_dimension_sdf_truth_prediction_error.pdf)

![DimensionとSDFの直接比較](22_formal_dimension_sdf_truth_prediction_error.png)

列は `COMSOL真値 | Dimension予測 | Dimension誤差 | SDF予測 | SDF誤差`、行は次の未知構造です。

- A：不等間隔
- B：高さ＋サイズ変化
- C：間隔＋高さ＋サイズ変化

物理量はイオン密度 `ni [10^17 m^-3]`、誤差は `予測 − COMSOL` です。全パネルで物理量の色軸と誤差の色軸を共通化しています。

## モデル別図

- [Formal Dimension（PNG）](20_formal_dimension_truth_prediction_error.png)／[SVG](20_formal_dimension_truth_prediction_error.svg)／[PDF](20_formal_dimension_truth_prediction_error.pdf)
- [Formal SDF（PNG）](20_formal_sdf_truth_prediction_error.png)／[SVG](20_formal_sdf_truth_prediction_error.svg)／[PDF](20_formal_sdf_truth_prediction_error.pdf)

## 数値と主張範囲

表示3例のplasma-domain相対L2は図中に記載しています。未知構造75例全体のイオン密度相対L2中央値は、Formal Dimension `27.37%`、Formal SDF `7.10%` です。

3例は空間的な誤差位置を説明する代表例であり、母集団全体の優劣は75例の集団統計で判断します。

- [表示ケースの数値CSV](spatial_case_metrics_v52.csv)
- [機械可読の検証結果](spatial_figure_validation_v52.json)

再生成は学習やCOMSOL計算を行わず、保存済み正式評価結果だけを描画します。

```powershell
.venv-torch\Scripts\python.exe experiments\icp_stage4\uno_epoch500_four_model_protocol_v47\plot_formal_dimension_sdf_truth_prediction_error_v52.py
```
