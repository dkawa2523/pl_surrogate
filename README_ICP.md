# ICP：コイル構造表現とサロゲート最適化ケース

## 目的

本検討では、ICPのコイル構造をサロゲートモデルへ与える方法を次の2系統で比較する。

1. **構造特徴量方式**：コイル形状をSDF・source場に変換し、空間チャンネルとして学習する。
2. **寸法パラメータ方式**：コイル寸法をスカラー条件として学習し、ケース依存のコイルSDFは入力しない。

両方式とも `ne`, `ni`, `Te`, `phi` の空間分布を推論し、その結果からウェハー近傍のBohmフラックス不均一性を最小化する。モデル間の予測を混合せず、各学習済みモデルで独立に最適化・評価する。

## ケース全体

| 区分 | 学習時の構造入力 | モデル | ケース名 | 状態 |
|---|---|---|---|---|
| 構造特徴量学習 | コイルunion SDF、加算source場 | U-Net | `icp_stage4_axisymmetric_energy_weighted_v1_unet` | 設定済み・未学習 |
| 構造特徴量学習 | コイルunion SDF、加算source場 | UNO | `icp_stage4_axisymmetric_energy_weighted_v1_uno` | 設定済み・未学習 |
| 構造特徴量最適化 | 各trialで構造特徴を再生成 | U-Net | `icp_stage4_simple_bohm_wafer_opt_v1_unet` | 学習後に実行 |
| 構造特徴量最適化 | 各trialで構造特徴を再生成 | UNO | `icp_stage4_simple_bohm_wafer_opt_v1_uno` | 学習後に実行 |
| 寸法学習 | 5構造寸法を条件ベクトル化 | U-Net | `icp_stage4_dimension_parameter_v1_unet` | 設定済み・未学習 |
| 寸法学習 | 5構造寸法を条件ベクトル化 | UNO | `icp_stage4_dimension_parameter_v1_uno` | 設定済み・未学習 |
| 寸法学習 | 5構造寸法＋各座標 | Coord-MLP | `icp_stage4_dimension_parameter_v1_coord_mlp` | 設定済み・未学習 |
| 寸法最適化 | 寸法条件を直接探索 | U-Net | `icp_stage4_dimension_parameter_bohm_opt_v1_unet` | 学習後に実行 |
| 寸法最適化 | 寸法条件を直接探索 | UNO | `icp_stage4_dimension_parameter_bohm_opt_v1_uno` | 学習後に実行 |
| 寸法最適化 | 寸法条件を直接探索 | Coord-MLP | `icp_stage4_dimension_parameter_bohm_opt_v1_coord_mlp` | 学習後に実行 |

「未学習」は、設定とrun出力先は用意されているが、有効な学習成果物がまだ生成されていないことを表す。

## 共通の学習条件

- データセット：`data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2`
- 出力：`ne`, `ni`, `Te`, `phi`
- 値変換：4物理量とも物理値のまま線形zスコア標準化
- zスコア統計：学習splitのプラズマ画素だけから算出
- clip：なし
- split：`structure_holdout`、seed 7、比率 0.8 / 0.1 / 0.1
- 学習seed：411
- epoch：70
- 点損失：ケース均等 `sample_mean` MSE
- 空間重み：軸対称体積に対応する半径重み
- `Te`：電子エネルギー分布を重視するため、物理値 `ne` による追加重み
- checkpoint：`best_val_loss`
- Physics Loss、勾配Loss、マルチスケールLoss：使用しない

入力表現以外を揃えることで、SDF方式と寸法方式の差を評価しやすくしている。

## 1. 構造特徴量による学習

### 入力

- プロセス条件：`pp`, `pp0`
- 空間チャンネル：`x`, `y`, `distance_signed`, `part_sdf_union`, `part_source_sum`
- 特徴プロファイル：`part_source_v1`
- `part_sdf_union`：全コイル形状のunion SDF
- `part_source_sum`：コイル数と近接効果を保持する滑らかな加算source場

部品番号ごとのSDFを並べず、コイル順序に依存しない表現とする。推論時にも学習時と同じ特徴生成処理を使用する。

### 学習ケース

- U-Net：[benchmark_icp_stage4_axisymmetric_energy_weighted_v1_unet.yaml](configs/experimental/icp_stage4/generated_axisymmetric_energy_weighted_v1/benchmark_icp_stage4_axisymmetric_energy_weighted_v1_unet.yaml)
- UNO：[benchmark_icp_stage4_axisymmetric_energy_weighted_v1_uno.yaml](configs/experimental/icp_stage4/generated_axisymmetric_energy_weighted_v1/benchmark_icp_stage4_axisymmetric_energy_weighted_v1_uno.yaml)

学習成果物の予定出力先：

- `runs/icp_stage4_axisymmetric_energy_weighted_v1/unet`
- `runs/icp_stage4_axisymmetric_energy_weighted_v1/uno`

## 2. 構造特徴量モデルによる最適化

### 問題設定

ウェハー近傍2セルにおけるBohmフラックスproxyを

`Gamma_B = ni * sqrt(Te)`

とし、軸対称面積重み付き変動係数を最小化する。

`objective = weighted_std(Gamma_B, r) / weighted_mean(Gamma_B, r)`

係数 `sqrt(e / m_i)` は変動係数と基準比では相殺されるため省略する。均一だがフラックスがほぼゼロの解を避けるため、面積平均フラックスに次の制約を置く。

`bohm_flux_area_mean >= 6.0249874e16`

これは基準ケース `case_g002_op01` の80%に相当する。

### 最適化変数

- プロセス：`pp`, `pp0`（データ分布の中央90%）
- 構造：`part.coil_01.tx` ～ `part.coil_06.tx`
- 各コイル半径方向shift：−0.015 ～ +0.015

各trialでparametric-parts providerがコイル形状と `part_source_v1` を再生成する。最適化器がSDF画素を直接編集する方式ではない。

### 最適化ケース

- U-Net：[benchmark_icp_stage4_simple_bohm_wafer_opt_v1_unet.yaml](configs/experimental/icp_stage4/generated_simple_bohm_wafer_opt_v1/benchmark_icp_stage4_simple_bohm_wafer_opt_v1_unet.yaml)
- UNO：[benchmark_icp_stage4_simple_bohm_wafer_opt_v1_uno.yaml](configs/experimental/icp_stage4/generated_simple_bohm_wafer_opt_v1/benchmark_icp_stage4_simple_bohm_wafer_opt_v1_uno.yaml)

Optuna TPE、192 trial、startup 32、seed 411とし、上位5解の空間分布を保存する。

## 3. 構造寸法パラメータによる学習

### 入力

- 構造寸法：`llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`
- プロセス条件：`pp`, `pp0`
- 条件ベクトル順：`llcoil, rrc, nncoil, rrce, zzc, pp, pp0`
- 固定空間チャンネル：`x`, `y`, `mask_plasma`, `distance_signed`, `distance_any`
- 特徴プロファイル：`geom_v1_mainline`

固定空間チャンネルはチャンバー座標とプラズマ領域をモデルへ伝えるためのものであり、ケース依存のコイルSDFやsource場は含まない。構造差は5つの寸法条件だけで表現する。

### 学習ケース

- U-Net：[benchmark_icp_stage4_dimension_parameter_v1_unet.yaml](configs/experimental/icp_stage4/generated_dimension_parameter_v1/benchmark_icp_stage4_dimension_parameter_v1_unet.yaml)
- UNO：[benchmark_icp_stage4_dimension_parameter_v1_uno.yaml](configs/experimental/icp_stage4/generated_dimension_parameter_v1/benchmark_icp_stage4_dimension_parameter_v1_uno.yaml)
- Coord-MLP：[benchmark_icp_stage4_dimension_parameter_v1_coord_mlp.yaml](configs/experimental/icp_stage4/generated_dimension_parameter_v1/benchmark_icp_stage4_dimension_parameter_v1_coord_mlp.yaml)

Coord-MLPは7条件と各空間座標から画素ごとの物理量を推論する。全格子を巨大な出力層で一括生成するGlobal MLPではない。

学習成果物の予定出力先：

- `runs/icp_stage4_dimension_parameter_v1/unet`
- `runs/icp_stage4_dimension_parameter_v1/uno`
- `runs/icp_stage4_dimension_parameter_v1/coord_mlp`

設定生成スクリプト：[generate_icp_stage4_dimension_parameter_v1.py](experiments/icp_stage4/scripts/generate_icp_stage4_dimension_parameter_v1.py)

## 4. 寸法学習モデルによる最適化

目的・制約・optimizerは構造特徴量方式と同一とする。一方、構造特徴画像は再生成せず、学習時の寸法条件を直接探索する。

### 探索範囲

| 変数 | 下限 | 上限 | 扱い |
|---|---:|---:|---|
| `llcoil` | 0.5457 | 1.4440 | 最適化 |
| `rrc` | 2.3979 | 9.5068 | 最適化 |
| `nncoil` | 4 | 4 | 固定 |
| `rrce` | 20.3595 | 29.3805 | 最適化 |
| `zzc` | 0.2423 | 4.6792 | 最適化 |
| `pp` | 586.464 | 2902.557 | 最適化 |
| `pp0` | 0.003807 | 0.086737 | 最適化 |

連続変数は外挿を抑えるためデータの5–95百分位に制限する。`nncoil`は整数変数だが、現行の共通optimizerは連続値を生成するため4巻に固定し、非物理的な小数巻数を避ける。

### 最適化ケース

- U-Net：[benchmark_icp_stage4_dimension_parameter_bohm_opt_v1_unet.yaml](configs/experimental/icp_stage4/generated_dimension_parameter_bohm_opt_v1/benchmark_icp_stage4_dimension_parameter_bohm_opt_v1_unet.yaml)
- UNO：[benchmark_icp_stage4_dimension_parameter_bohm_opt_v1_uno.yaml](configs/experimental/icp_stage4/generated_dimension_parameter_bohm_opt_v1/benchmark_icp_stage4_dimension_parameter_bohm_opt_v1_uno.yaml)
- Coord-MLP：[benchmark_icp_stage4_dimension_parameter_bohm_opt_v1_coord_mlp.yaml](configs/experimental/icp_stage4/generated_dimension_parameter_bohm_opt_v1/benchmark_icp_stage4_dimension_parameter_bohm_opt_v1_coord_mlp.yaml)

## 実行順序

1. 構造特徴量U-Net・UNOをそれぞれ学習する。
2. 各モデルの空間分布精度とウェハー近傍 `ni`, `Te` を評価する。
3. 品質を満たしたモデルだけで構造特徴量最適化を実行する。
4. 寸法条件U-Net・UNO・Coord-MLPをそれぞれ学習する。
5. 同じ評価を行い、品質を満たしたモデルだけで寸法最適化を実行する。
6. 同一モデル内で最適化前後を比較し、その後に入力表現間・モデル間を比較する。

学習コマンド例：

```powershell
.venv-torch\Scripts\plasma-surrogate.exe benchmark run --config <training-config.yaml>
```

学習済みモデルを使う最適化コマンド例：

```powershell
.venv-torch\Scripts\plasma-surrogate.exe infer --config <optimization-config.yaml>
```

## 比較時に必ず確認する指標

- 各物理量・各ケースのphysical relative L2
- 勾配relative L2と空間分布図
- median、p90、worstケース
- ウェハー近傍BohmフラックスCV
- ウェハー近傍Bohmフラックス面積平均
- 最適化前後の `ne`, `ni`, `Te`, `phi` 空間分布
- 最適解が学習条件範囲内にあるか
- U-Net、UNO、Coord-MLP間で最適解が一致するか、または大きく乖離するか

最適化値だけが良くても、基礎となる `ni` と `Te` の空間予測が悪ければ最適解は採用しない。Lossや全画素一括スコアではなく、ケース別空間評価を先に合格条件とする。

## 2方式の解釈

| 観点 | 構造特徴量方式 | 寸法パラメータ方式 |
|---|---|---|
| 構造表現 | 空間SDF・source場 | 5つの寸法スカラー |
| 未知形状への拡張 | 表現可能だが学習範囲の制約あり | 定義済み寸法族の範囲内 |
| 最適化変数 | 形状providerの幾何変数 | 学習条件と同じ寸法変数 |
| trial時の特徴生成 | 必要 | 不要 |
| 計算と実装 | やや重い | 単純で高速 |
| 主な用途 | 将来の形状生成・形状最適化 | 寸法設計と説明性の高い対照実験 |

寸法方式が同等精度なら、単純さと最適化安定性の面で寸法方式が有利である。構造特徴量方式が明確に高精度なら、SDFが寸法ベクトルでは表現できない局所的な形状効果を学習していると判断できる。

## 既存runとの区別

`runs/icp_stage4_coil_structure_v1` は、以前のlog変換・空間複合Lossを用いた実行済み検証runであり、本READMEの線形物理値比較設定とは別系列である。過去結果の参照には利用できるが、今回のSDF対寸法比較の学習済みモデルとして混用しない。
