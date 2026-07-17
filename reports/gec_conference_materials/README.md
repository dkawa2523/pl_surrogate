# GEC-CCP 学会発表用・採用結果 README

このREADMEは、GEC-CCPサロゲートモデル比較で学会発表に採用する図、評価条件、学習lossを
第三者が同じ意味で読めるように固定した入口です。画像の正本は
[`training_size_ablation/`](training_size_ablation/index.md)に保存し、このページから用途順に参照します。

## 採用図

| 順序 | 図の目的 | 横軸・列 | 縦軸・行 | ファイル |
|---:|---|---|---|---|
| 1 | 電子系物性のモデル比較 | 電子密度 case-macro relative RMSE | 電子温度 case-macro relative RMSE | [SVG](training_size_ablation/ccp_n54_relative_rmse_ne_vs_te.svg) / [PNG](training_size_ablation/ccp_n54_relative_rmse_ne_vs_te.png) / [PDF](training_size_ablation/ccp_n54_relative_rmse_ne_vs_te.pdf) |
| 2 | 電位・イオン系物性のモデル比較 | 電位 case-macro relative RMSE | イオン密度 case-macro relative RMSE | [SVG](training_size_ablation/ccp_n54_relative_rmse_phi_vs_ni.svg) / [PNG](training_size_ablation/ccp_n54_relative_rmse_phi_vs_ni.png) / [PDF](training_size_ablation/ccp_n54_relative_rmse_phi_vs_ni.pdf) |
| 3 | 電子密度場の定性的・局所的比較 | Truth / Prediction / Signed error | FFNO / POD-DeepONet / DenseMLP | [SVG](training_size_ablation/ccp_n54_ffno_pod_deeponet_densemlp_ne_comparison.svg) / [PNG](training_size_ablation/ccp_n54_ffno_pod_deeponet_densemlp_ne_comparison.png) / [PDF](training_size_ablation/ccp_n54_ffno_pod_deeponet_densemlp_ne_comparison.pdf) |

機械可読の採用一覧は[adopted_figures.csv](adopted_figures.csv)、RMSE値は
[pair scatter data](training_size_ablation/ccp_n54_relative_rmse_pair_scatter_data.csv)にあります。

## 評価条件

| 項目 | 内容 |
|---|---|
| データセット | GEC-CCP 78条件 |
| split | Train 54 / Validation 11 / Test 13 |
| 図の学習条件数 | 54 |
| 評価test | 固定した13件の補間testケース |
| 学習seed | 412 |
| モデル選択 | Validation-only |
| RMSE散布図から除外 | Global MLP、U-NO、DeepONet Plasma |
| `DeepONet`の意味 | branch調整済みPOD-DeepONet |
| 空間分布の共通ケース | `PP0=3, Td=0.03, gamma=0.04, PA=0.1` |

各物性の散布図は、生のRMSEではなく物性間で比較可能なcase-macro relative RMSEを使います。
物性を (k)、testケースを (c)、plasma画素集合を \(\Omega_c\) とすると、

\[
\operatorname{rRMSE}_{c,k}=
\frac{\sqrt{\frac{1}{|\Omega_c|}\sum_{p\in\Omega_c}
(\hat y_{c,k,p}-y_{c,k,p})^2}}
{\sqrt{\frac{1}{|\Omega_c|}\sum_{p\in\Omega_c}y_{c,k,p}^2}},
\qquad
E_k=\frac{100}{13}\sum_{c=1}^{13}\operatorname{rRMSE}_{c,k}.
\]

散布図の各座標は物理単位へ逆変換した後の \(E_k\) [%] です。左下ほど高精度です。

空間誤差図は、同じtestケースの電子密度ピークで正規化した符号付き誤差です。

\[
e_{c,p}^{\mathrm{map}}=
100\,\frac{\hat n_{e,c,p}-n_{e,c,p}}
{\max_{q\in\Omega_c}|n_{e,c,q}|}.
\]

全モデル共通のカラーレンジを \(\pm10\%\) とし、赤は過大予測、青は過小予測を表します。

## 学習loss

### 1. 学習targetの変換

4物性 \(k\in\{n_e,n_i,T_e,\phi\}\) は対数化せず、物理値にidentity変換を適用した後、
Train 54ケースのplasma画素だけから求めた平均と標準偏差で線形Z-score化します。

\[
\tilde y_{c,k,p}=\frac{y_{c,k,p}-\mu_k^{\mathrm{train,plasma}}}
{\sigma_k^{\mathrm{train,plasma}}},
\qquad
r_{c,k,p}=\hat{\tilde y}_{c,k,p}-\tilde y_{c,k,p}.
\]

clip、対数変換、分位点変換は使いません。以下のlossはこのZ-score空間で計算します。

### 2. Huber関数

\(\delta=1\) のHuber lossを使用します。

\[
H_1(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\le 1,\\
|r|-\frac{1}{2}, & |r|>1.
\end{cases}
\]

### 3. 物性ごとの空間loss

各ケースを同じ重みで平均する`sample_mean`を使い、plasma領域だけで次を計算します。

\[
\mathcal L_k=
\mathcal L_{\mathrm{point},k}
+0.25\,\mathcal L_{\mathrm{boundary},k}
+0.10\,\mathcal L_{\mathrm{gradient},k}
+0.05\,\mathcal L_{\mathrm{multiscale},k}.
\]

- \(\mathcal L_{\mathrm{point},k}\): plasma全画素の \(H_1(r)\) をケース内平均し、batch内で平均。
- \(\mathcal L_{\mathrm{boundary},k}\): plasma境界から2 pixel以内の同じHuber loss。
- \(\mathcal L_{\mathrm{gradient},k}\): \(r,z\)方向の隣接差分について、予測勾配と真値勾配の差へHuberを適用。spacingは`[1, 1]` pixelで、2方向を平均。
- \(\mathcal L_{\mathrm{multiscale},k}\): plasma mask付き平均poolingをscale 2、4で行い、各解像度のHuber lossを平均。

勾配項は、隣接する両画素がplasma領域にあるedgeだけを使用します。境界外への人工的な値飛びは
lossへ含めません。

#### 電子密度で書いたpoint lossとgradient loss

電子密度のZ-score真値を \(\tilde n_{e,c,i,j}\)、予測値を
\(\hat{\tilde n}_{e,c,i,j}\)、plasma内で1となるmaskを \(M_{c,i,j}\) とします。
電子密度のpoint lossは

\[
\boxed{\mathcal L_{\mathrm{point},n_e}
=\frac{1}{B}\sum_{c=1}^{B}
\frac{
\sum_{i,j}M_{c,i,j}
H_1\!\left(\hat{\tilde n}_{e,c,i,j}-\tilde n_{e,c,i,j}\right)}
{\sum_{i,j}M_{c,i,j}}}
\]

です。勾配項では、差分の両端がplasma内にある場合だけ1となるedge maskを

\[
M^{(z)}_{c,i,j}=M_{c,i+1,j}M_{c,i,j},\qquad
M^{(r)}_{c,i,j}=M_{c,i,j+1}M_{c,i,j}
\]

とします。実学習設定のpixel spacingは \(\Delta z=\Delta r=1\) なので、

\[
\varepsilon^{(z)}_{c,i,j}=
\left(\hat{\tilde n}_{e,c,i+1,j}-\hat{\tilde n}_{e,c,i,j}\right)
-\left(\tilde n_{e,c,i+1,j}-\tilde n_{e,c,i,j}\right),
\]

\[
\varepsilon^{(r)}_{c,i,j}=
\left(\hat{\tilde n}_{e,c,i,j+1}-\hat{\tilde n}_{e,c,i,j}\right)
-\left(\tilde n_{e,c,i,j+1}-\tilde n_{e,c,i,j}\right)
\]

となり、電子密度のgradient lossは

\[
\boxed{
\mathcal L_{\mathrm{gradient},n_e}
=\frac{1}{2B}\sum_{c=1}^{B}
\left[
\frac{\sum_{i,j}M^{(z)}_{c,i,j}H_1\!\left(\varepsilon^{(z)}_{c,i,j}\right)}
{\sum_{i,j}M^{(z)}_{c,i,j}}
+
\frac{\sum_{i,j}M^{(r)}_{c,i,j}H_1\!\left(\varepsilon^{(r)}_{c,i,j}\right)}
{\sum_{i,j}M^{(r)}_{c,i,j}}
\right]}
\]

です。すなわち、電子密度そのものの差ではなく、隣接画素間の電子密度変化量の差を
Huber lossで評価しています。

### 4. 物性familyの均等化

物性familyを密度 \(\{n_e,n_i\}\)、温度 \(\{T_e\}\)、静電ポテンシャル \(\{\phi\}\) の3群に分け、
各familyへ \(1/3\) を割り当てます。密度familyだけ2物性で等分するため、最終的な共通field lossは

\[
\boxed{
\mathcal L_{\mathrm{field}}=
\frac{1}{6}\mathcal L_{n_e}
+\frac{1}{6}\mathcal L_{n_i}
+\frac{1}{3}\mathcal L_{T_e}
+\frac{1}{3}\mathcal L_{\phi}}
\]

です。FFNO、FNO、DenseMLP、ResMLP、U-Net、U-Net++、CNOではこの共通式を学習data lossとして
使用します。Poisson残差などのphysics lossは本比較では無効です。

### 5. POD-DeepONetだけの補助loss

採用したPOD-DeepONetは、共通field lossにPOD係数推定の補助MSEを追加します。

\[
\mathcal L_{\mathrm{DeepONet}}=
\mathcal L_{\mathrm{field}}+0.02\,\mathcal L_{\mathrm{coeff}},
\qquad
\mathcal L_{\mathrm{coeff}}=
\sum_k w_k\,\frac{1}{B R_k}
\sum_{c=1}^{B}\sum_{j=1}^{R_k}
(\hat a_{c,k,j}-a_{c,k,j})^2.
\]

ここで \(R_k\) は物性 \(k\) のPOD rank、\(w_k\) はfield lossと同じ
\((1/6,1/6,1/3,1/3)\) です。係数数の多い物性が自動的に支配しないよう、物性内で先に平均します。

### 6. 学習lossと発表指標の違い

学習はZ-score空間のHuber複合loss、発表図は物理値へ戻したrelative RMSEです。したがって、
loss値と散布図のRMSE値は同じ数値ではありません。checkpointはValidation-onlyの空間目的関数で
選び、Test 13ケースは最終評価にだけ使用します。

lossの計算手順図は[PNG](gec_ccp_loss_calculation.png) / [PDF](gec_ccp_loss_calculation.pdf) /
[SVG](gec_ccp_loss_calculation.svg)、詳細値は
[metadata](gec_ccp_loss_calculation_metadata.json)にあります。

## 再生成とprovenance

```powershell
.\.venv-torch\Scripts\python.exe experiments/conference/scripts/plot_gec_ccp_n54_fieldwise_rmse.py
$env:PYTHONPATH='src;scripts'
.\.venv-torch\Scripts\python.exe scripts/plot_gec_ccp_n54_model_density_comparison.py
```

- RMSE散布図生成元: [plot_gec_ccp_n54_fieldwise_rmse.py](../../experiments/conference/scripts/plot_gec_ccp_n54_fieldwise_rmse.py)
- 空間分布生成元: [plot_gec_ccp_n54_model_density_comparison.py](../../scripts/plot_gec_ccp_n54_model_density_comparison.py)
- RMSE図metadata: [ccp_n54_relative_rmse_pair_scatter_metadata.json](training_size_ablation/ccp_n54_relative_rmse_pair_scatter_metadata.json)
- 空間図metadata: [ccp_n54_ffno_pod_deeponet_densemlp_ne_comparison_metadata.json](training_size_ablation/ccp_n54_ffno_pod_deeponet_densemlp_ne_comparison_metadata.json)
- loss実装: [loss_composer.py](../../src/plasma_surrogate/train/loss_composer.py)
