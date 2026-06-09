# ICDDPS招待講演 全体ストーリー整理メモ

## 1. 発表の位置づけ

本講演は、プラズマサロゲートモデルの単なる技術紹介ではなく、半導体製造企業として、プラズマ×機械学習をどのように産業開発の基盤技術へ引き上げるかを示す招待講演として内容検討。

主題は、特定モデルの精度競争ではなく、半導体製造において実際に使えるプラズマサロゲートとは何か、どのような物理量の扱い、評価指標、モデル選定、検証ループが必要かを、公開標準系と産業応用の二段構成で説明する。

発表全体の中心メッセージは以下である。

> 半導体製造に使えるプラズマサロゲートは、単なる高速な深層学習モデルではない。  
> プラズマ物性場を物理的に意味のあるtargetとして扱い、境界・バルク・wafer近傍で信頼性を評価し、さらに装置構造最適化へ接続できる、物理監査可能な設計支援基盤である。

この立場を明確にすることで、プラズマ物理研究者、半導体関係者のそれぞれに対して、異なる価値を提示する構成。

---

## 2. 想定聴衆と刺さる論点

### 2.1 プラズマ物理研究者

プラズマ物理研究者に対しては、平均的なR²やモデルランキングだけを示しても説得力は弱い。重要なのは、機械学習がプラズマ物性の意味を壊していないこと、また、どこで失敗するかを物理的に解釈できること。

強調すべき点は以下である。

- 電子密度、イオン密度、電子温度、電位を単なる画像チャンネルとして扱わない。
- 密度場は単純にlog表現でなく、密度の連続性が重要であること密度分布からZスコアで標準化して、学習することがよい。たんじゅに対数スケールで学習してもうまくいかない例。
- 電位は値そのものだけでなく、勾配、sheath、境界近傍での誤差が重要である。
- global R²ではなく、bulk、boundary、deep plasma、wafer-near、edgeなどの領域別評価を行う。
- negative ratio、準中性proxy、Poisson residual proxy、電位勾配誤差などを、物理診断指標として用いる。
- サロゲートは物理シミュレーションの置き換えではなく、高忠実計算へ戻す候補生成器として使う。

### 2.2 半導体分野の企業エンジニア

半導体プロセス・装置エンジニアにとって重要なのは、場全体の平均精度ではなく、プロセス判断に効く場所と量である。

強調すべき点は以下である。

- wafer面のイオンフラックス、均一性、center-edge ratio、edge挙動が重要である。
- chamber、coil、window、wafer、biasなどの構造・条件変更が、電力吸収、密度分布、wafer QoIにどうつながるかを可視化する。
- サロゲートにより、数千〜数万条件の仮想スクリーニングが可能になる。
- ただし最終判断は高忠実シミュレーション、必要に応じて実験検証へ戻す。
- 企業としての価値は、単にモデルを持つことではなく、開発判断に使える検証可能なワークフローを持つことである。





---

## 4. 30分講演の全体構成

| 時間 | パート | 目的 |
|---:|---|---|
| 0:00–3:00 | 導入：なぜ半導体製造にプラズマサロゲートが必要か | 企業視点の問題設定を示す |
| 3:00–6:00 | 産業応用の全体像 | process window、virtual experiment、装置設計への接続 |
| 6:00–10:00 | Neural NetworkとNeural Operatorの違い | 深層学習の技術的土台を整理 |
| 10:00–13:00 | リポジトリ基盤とdata contract | 研究コードではなく開発基盤としての価値を示す |
| 13:00–21:00 | GEC-CCP：field trustworthiness | target設計、密度前処理、領域別評価、POD系の論理 |
| 21:00–28:00 | ICPコイル構造最適化 | U-Net family / FFNOによるgeometry-aware design |
| 28:00–30:00 | 結論 | 企業としてのメッセージを締める |

---

## 5. 発表ストーリーの一本線

発表全体は、次の論理でつなぐ。

```text
半導体製造では、wafer近傍、sheath、edge、uniformityがプロセス判断に効く
        ↓
したがって、プラズマサロゲートはglobal R²だけでは評価できない
        ↓
まずGEC-CCPで、table condition → plasma field の信頼性を定義する
        ↓
密度・温度・電位を物理量として扱い、
plasma_only scaling、領域別評価、物理diagnosticsを導入する
        ↓
固定形状fieldではPOD系が自然で、
データ増加時にはPOD + coordinate residualが局所補正に有効
        ↓
一方、ICP coil設計ではgeometryが出力に効くため、
U-Net とFFNOを使い分ける
        ↓
U-Net は局所構造、
FFNOは大域的・非局所的なfield responseを見る
        ↓
wafer 歩留まりにきく均一性を目的関数としてcoil構造候補を探索する
        ↓
サロゲートは最終判断者ではなく、
高忠実シミュレーションへ戻す候補生成器である
        ↓
プラズマ×AIを、半導体製造の産業設計基盤として提示する
```

---

## 6. Neural NetworkとNeural Operatorの説明方針

この講演では、Neural NetworkとNeural Operatorを単純な優劣ではなく、学習対象の違いとして説明する。

### 6.1 Neural Network

一般的なNeural Networkは、有限次元の写像を学習する。

\[
f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^m
\]

たとえば、プロセス条件ベクトルから固定格子上の2D fieldを出すGlobal MLPは、以下のように表せる。

\[
\hat{u} = f_\theta(\mathbf{c})
\]

ここで、\(\mathbf{c}\)は圧力、RF power、gas、biasなどの条件ベクトル、\(\hat{u}\)は固定格子上のプラズマ物性場である。

U-Netは、固定grid上の局所構造を扱う畳み込み型のモデルである。skip connectionにより、局所的な境界、壁、wafer近傍、coil/window近傍の特徴を保持しやすい。

### 6.2 Neural Operator

Neural Operatorは、有限次元ベクトルではなく、関数空間間の写像を近似する枠組みである。

\[
\mathcal{G}_\theta: \mathcal{A}(\Omega) \rightarrow \mathcal{U}(\Omega)
\]

これは、PDEやプラズマシミュレーションが本質的に持つ「条件・境界・形状から場全体を決める解作用素」を学習する考え方に近い。

FNO/FFNOでは、Fourier空間でglobalな結合を扱う。ICPのように、coil構造、誘導電場、電力吸収、密度分布が非局所的に結合する問題では、Neural Operator的な考え方が自然である。

### 6.3 講演での結論

Neural NetworkとNeural Operatorは競合ではない。問題構造に応じて使い分ける。

| 問題 | 向くモデル | 理由 |
|---|---|---|
| 固定形状、条件だけ変える | Global MLP / POD / POD-DeepONet | 大域応答を少数モードで表現しやすい |
| 境界・局所構造が効く | U-Net family | 畳み込みとskip connectionにより局所特徴を拾いやすい |
| 非局所応答が強い | FNO / FFNO | Fourier空間で大域結合を扱いやすい |
| 大域構造に局所補正を足したい | Coord MLP POD residual | POD基底外の局所誤差を補正できる |

---

## 7. GEC-CCPパートの詳細

### 7.1 GEC-CCPの役割

GEC-CCPは、講演の主役ではなく、プラズマ物性場サロゲートの設計原理を示すための公開標準系として使う。

GEC-CCPでは、入力はテーブル条件であり、出力は固定格子上の2次元プラズマ物性場である。

\[
\mathcal{S}^{\mathrm{CCP}}_\theta:
\mathbf{c}
\mapsto
[n_e(r,z), n_i(r,z), T_e(r,z), \phi(r,z)]
\]

ここで重要なのは、geometryを最適化対象として入力するのではなく、固定形状の中で、条件変化に対する場の応答を学習する点である。

このため、GEC-CCPは `table_only` の問題として整理する。

### 7.2 密度targetの扱い

プラズマ密度のようなデータは `log_ne`、`log_ni` として学習されている場合がデータサイエンス的観点。ただし、これは学習に適さない。


密度を物理密度として扱う利点は以下である。

- wafer flux proxyに接続しやすい。
- bulk密度の物理的な意味を保てる。
- 非負性を直接診断できる。
- 逆変換後の誤差を物理単位で解釈しやすい。

講演では、以下の一文を明確に述べる。

> 密度はsource側でlogとして保存されていても、学習targetとしてlog-densityを学習しているわけではない。物理密度へ戻した上で、plasma領域で標準化して扱う。

### 7.3 物性ごとのfailure mode

各targetには異なる物理的意味と失敗モードがある。

| target | 物理的意味 | 典型的な失敗 | 評価すべき領域 |
|---|---|---|---|
| \(n_e\) | plasma生成、電離、bulk強度 | bulkは合うがedgeで外れる | bulk, edge, non-negativity |
| \(n_i\) | ion flux proxy、sheath近傍 | \(n_e\)との整合が崩れる | wafer-near, quasi-neutrality |
| \(T_e\) | 電離・解離反応 | deepや局所ピークを外す | deep plasma, local peak |
| \(\phi\) | 電場、sheath、イオン加速 | 値は合うが勾配が外れる | boundary, sheath proxy, \(\nabla\phi\) |

この説明により、プラズマfieldを単なる4チャンネル画像として扱っていないことを示す。

### 7.4 領域別評価

global R²だけでは、半導体用途で必要な信頼性は判断できない。理由は、同じ誤差でも場所により意味が異なるためである。

評価領域は以下のように分ける。

| 領域 | 物理的意味 | 半導体用途での意味 |
|---|---|---|
| bulk plasma | 準中性、主生成領域 | 反応種供給、全体密度 |
| deep plasma | 境界から離れた内部領域 | 安定したbulk状態の把握 |
| boundary / sheath proxy | 電位降下、壁・電極近傍 | イオン加速、表面反応 |
| wafer-near proxy | wafer直上 | etch/deposition均一性 |
| edge region | 端部、壁損失 | edge uniformity、chamber matching |

代表的な指標は以下である。

\[
R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}
\]

\[
R^2_{\mathrm{dual}} = \frac{1}{2}(R^2_{\mathrm{interp}} + R^2_{\mathrm{extrap}})
\]

\[
\mathrm{negative\ ratio} = \frac{\#\{i \mid \hat{y}_i < 0\}}{N}
\]

\[
\Delta R^2_{\mathrm{boundary-deep}}
= R^2_{\mathrm{boundary}} - R^2_{\mathrm{deep}}
\]

### 7.5 GEC-CCPでのモデル選定

GEC-CCPは固定形状・table condition入力であるため、まずPOD系が自然な候補になる。

固定形状fieldは、以下のように低次元モード展開できる。

\[
u(\mathbf{x};\mathbf{c})
\approx
\bar{u}(\mathbf{x})
+
\sum_{k=1}^{K} a_k(\mathbf{c})\psi_k(\mathbf{x})
\]

ここで、\(\psi_k(\mathbf{x})\)はPOD mode、\(a_k(\mathbf{c})\)は条件依存係数である。

POD-DeepONet系では、この係数を条件から予測する。

\[
a_k = f_{\theta,k}(\mathbf{c})
\]

一方、Coord MLP POD residualでは、PODで捉えきれない局所構造を座標依存残差として補正する。

\[
\hat{u}(\mathbf{x};\mathbf{c})
=
\bar{u}(\mathbf{x})
+
\sum_{k=1}^{K} a_k(\mathbf{c})\psi_k(\mathbf{x})
+
r_\theta(\mathbf{c},\mathbf{x})
\]

このモデルは、globalなプラズマ応答はPODで捉え、boundaryやsheath-like regionの局所誤差をresidualで補う、という物理的に説明しやすい構造を持つ。

### 7.6 結果の解釈

27/54/78ケース比較では、POD系、FNO/FFNOが有力候補として整理されている。

講演では、数値ランキングそのものより、以下の解釈を示す。

| 結果 | 解釈 |
|---|---|
| 少数ケースではPOD系が強い | 固定形状fieldでは低次元大域モードが有効 |
| 78ケースでCoord MLP POD residualが上位 | 大域POD + 局所補正の構造が有効 |
| FNO/FFNOが安定して上位 | 非局所応答を扱うoperator型モデルは重要な比較軸 |
| U-Net familyはplain U-Netより拡張版が有効 | 局所構造には有効だが、設計問題での使い方が重要 |

ここでの結論は、以下である。

> モデル選定は、ランキングではなく問題構造に従うべきである。固定形状・table conditionではPOD系が自然であり、構造依存・非局所応答ではU-Net familyやFNO/FFNOが重要になる。

---

## 8. ICPコイル構造最適化パートの詳細

### 8.1 ICPパートの役割

ICPパートでは、前半で示したfield trustworthinessの考え方を、実際の装置構造最適化へ拡張する。

ICPでは、coil構造が誘導電場、電力吸収、密度分布、wafer fluxに影響する。このため、入力は条件だけでなくgeometryを含む。

\[
\mathcal{S}^{\mathrm{ICP}}_\theta:
(\mathbf{c},\mathbf{g})
\mapsto
[n_e, T_e, \phi, P_{\mathrm{abs}}, \Gamma_i, \mathrm{QoIs}]
\]

ここで、\(\mathbf{g}\)はcoil半径、coil高さ、pitch、turn数、window、chamberなどの設計変数である。

### 8.2 raw pixel optimizationではなく、意味ある設計変数を最適化する

企業発表として重要なのは、AIにreactor形状を自由に描かせるのではないという点である。

最適化するのは、製造・設計上意味のある低次元のgeometry parameterである。

例：

\[
\mathbf{g} = [R_{\mathrm{in}}, R_{\mathrm{out}}, H, N_{\mathrm{turn}}, pitch, width, z_{\mathrm{coil}}, \ldots]
\]

この設計変数から、SDFやgeometry channelを生成し、U-Net familyやFFNOへ入力する。

講演では次の言い方がよい。

> AIに装置を自由生成させるのではなく、設計制約を満たす意味あるgeometry parameterの範囲で探索する。

### 8.3 U-Net familyとFFNOの役割分担

ICPでは、U-Net familyとFFNOを競合モデルではなく、相補的なモデルとして扱う。

| モデル | 役割 | ICPでの意味 |
|---|---|---|
| U-Net family | 局所構造・境界・window/wafer近傍 | coilやwall近傍の局所変化に敏感 |
| U-Net++ / U-Net Operator系 | multi-scale局所特徴 | plain U-Netより安定した局所場予測 |
| FFNO | Fourier空間での大域応答 | 誘導電場、電力吸収、密度分布の非局所結合 |
| U-Net family + FFNO | 予測一致/不一致を見る | 一致候補は有望、不一致候補は再シミュレーション対象 |

### 8.4 最適化目的関数

ICPコイル最適化では、密度最大化ではなくwafer QoIを目的関数にする。

\[
J(\mathbf{g})
=
w_1
\frac{\sigma(\Gamma_i)}{\overline{\Gamma_i}}
+
w_2
\left|
\frac{\overline{\Gamma_i}-\Gamma_i^\ast}{\Gamma_i^\ast}
\right|
+
w_3 R_{\mathrm{phys}}
+
w_4 C_{\mathrm{eng}}
\]

各項の意味は以下である。

| 項 | 意味 |
|---|---|
| 第1項 | wafer面flux non-uniformity |
| 第2項 | 目標fluxからのズレ |
| 第3項 | 物理診断指標、negative ratio、boundary/deep errorなど |
| 第4項 | coil寸法、熱、電流、製造制約など |

講演での重要な一文は以下である。

> 目的はプラズマ密度をどこでも最大化することではなく、wafer levelのプロセス性能に効くようにプラズマ応答を整形することである。

### 8.5 ICP結果スライドの見せ方

ICP結果は、以下の因果関係で示す。

```text
coil geometry
→ induced field / power absorption
→ plasma density distribution
→ wafer flux profile
→ uniformity / process window
```


### 8.6 検証ループ

サロゲート最適化は最終判断ではない。最終的には高忠実シミュレーションへ戻す。

```text
Surrogate screening over many coil candidates
        ↓
Select top candidates using QoI + physics diagnostics
        ↓
High-fidelity plasma simulation
        ↓
Accept / reject / refine
```

講演では、以下の一文を必ず入れる。

> サロゲートは探索空間を狭めるためのproposal generatorであり、最終判断者ではない。

-
---

## 最終メッセージ

最後は、以下のメッセージで締める。

> 我々の目的は、プラズマ物理を機械学習で置き換えることではない。  
> プラズマシミュレーションを、より速く、より監査可能に、そして半導体プロセス・装置設計の意思決定へ直接つながる技術へ引き上げることである。

