# ICP UNO 4モデル・500 epoch可搬実行仕様 v47

## この文書の位置づけ

本ディレクトリは、次の4モデルを他環境で再学習・評価・最適化し、
学会用グラフまで作成するための専用仕様書です。

本ブランチでは、固定データセット
`data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/` を Git LFS で管理します。
別環境では checkout 後に `git lfs install` と
`git lfs pull --include="data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/**"`
を実行し、配列実体を取得してから検証・学習してください。

他環境での入口、取得対象、実装元の対応は
[REMOTE_TRAINING_START_HERE_JA.md](REMOTE_TRAINING_START_HERE_JA.md)を最初に参照してください。
本ブランチにはデータに加えて、現行Python source snapshot、正式設定・初期重み、
ABC-SDF参照実装・初期重み、正式参照checkpointを含めます。

目的別の直接リンクは次のとおりです。

- エージェントへ貼る指示文：
  [REMOTE_AGENT_500EPOCH_PROMPT_JA.md](REMOTE_AGENT_500EPOCH_PROMPT_JA.md)
- データセット・Git LFS取得：
  [DATASET_ACQUISITION_JA.md](DATASET_ACQUISITION_JA.md)
- 500 epoch学習・resume仕様：
  [docs/03_TRAINING_RUNBOOK.md](docs/03_TRAINING_RUNBOOK.md)
- 評価・最適化仕様：
  [docs/04_EVALUATION_AND_OPTIMIZATION.md](docs/04_EVALUATION_AND_OPTIMIZATION.md)
- 学会用図仕様：
  [docs/05_CONFERENCE_FIGURES.md](docs/05_CONFERENCE_FIGURES.md)
- 監査・完了条件：
  [docs/06_RECOVERY_AUDIT_ACCEPTANCE.md](docs/06_RECOVERY_AUDIT_ACCEPTANCE.md)

1. 正式SDF
2. 正式Dimension
3. ABC-SDF：多重解像度、入力群別lifting、適応的局所・大域混合
4. ABC-Dimension-P：ABC-SDFと比較可能な寸法パラメータ対照モデル

既存のv45正式モデルやv46検証モデルへ説明を追記せず、次の専用領域へ
完全に分離しています。

```text
experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/
runs/icp_uno_epoch500_four_model_v47/
reports/icp_conference_materials/icp_uno_epoch500_four_model_v47/
```

既存の次の領域は参照専用であり、上書き禁止です。

```text
experiments/icp_stage4/conference_continuity_v45/
experiments/icp_stage4/uno_structure_em_v46/
runs/icp_conference_continuity_v45/
runs/icp_uno_structure_em_v46/
```

## 現在の状態

本タスクでは実行仕様、比較条件、環境移行方法、最適化計画、図仕様、
監査方法を固定しました。500 epochの学習自体はまだ開始していません。
したがって、本ディレクトリに500 epochの精度結果は記載していません。

実行コードを作成する際は、既存コードへ直接追加せず、本ディレクトリに
隔離ランナーを実装します。長時間実行前に、optimizer、cosine schedule、
乱数状態まで含む途中再開試験を通す必要があります。

## 4モデルの比較関係

正式モデルの比較とABCモデルの比較は分けて解釈します。

- 正式SDF対正式Dimension：現在の正式表現の比較
- ABC-SDF対ABC-Dimension-P：ABC骨格と磁場考慮を揃えた表現比較

正式モデルとABCモデルを直接比較する場合は、骨格と入力チャネルの違いを
含む総合比較であり、SDF表現だけの効果とは解釈しません。

### 正式SDF

- 条件：`pp`, `pp0`
- 空間入力：座標、プラズママスク、壁距離、コイルunion SDF、平均源分布
- 真空磁場入力：なし
- 正式初期重みを使用

### 正式Dimension

- 条件：`llcoil`, `rrc`, `nncoil`, `rrce`, `zzc`, `pp`, `pp0`
- 空間入力：座標、プラズママスク、壁距離
- 真空磁場入力：なし
- 不規則構造は5寸法への投影値で表現
- 正式初期重みを使用

### ABC-SDF

- 条件：`pp`, `pp0`
- 静的空間入力5ch
- 実構造入力5ch
- 実構造から決定論的に計算した単位電流真空磁場4ch
- 多重解像度UNO
- 条件・静的空間・構造・磁場の個別lifting
- 各位置で局所畳み込みと大域Fourier演算を混合する学習ゲート

磁場はCOMSOLのプラズマ応答ではなく、コイル構造から計算した
`Aphi`, `Br`, `Bz`, `|B|`です。目的変数は`ne`, `ni`, `Te`, `phi`のままです。

### ABC-Dimension-P

ABC-SDFとの比較を成立させるため、単純な「7変数＋ABC」にはしません。

- 運転条件は`pp`, `pp0`の2ch
- 5寸法を正規化し、空間全体へbroadcastした5chを構造群として入力
- 5寸法から等間隔・同一高さ・同一サイズの規則構造を再構成
- 再構成した規則構造から単位電流真空磁場4chを計算
- ABC-SDFと同じ静的空間5ch、同じUNO骨格、同じゲート、同じ出力head

重要なのは、Dimension側の磁場を実際の不規則構造から計算しないことです。
実構造由来磁場を入力すると、寸法変数にない個別配置情報が磁場経由で入り、
純粋な寸法対照モデルではなくなります。

ABC-SDFとABC-Dimension-Pは、条件2ch、空間14ch、trainable parameter数を
一致させます。parameter数が一致しなければ比較契約違反です。

## 学習条件

- データセット：957件
- split：train 685、validation 136、test 136
- 学習seed：1237
- epoch：各500、early stoppingなし
- batch size：16
- optimizer：AdamW
- learning rate：`3e-4`
- weight decay：`1e-4`
- schedule：500 epoch全体に対するcosine
- 目的変数：`ne`, `ni`, `Te`, `phi`
- 損失、target scaling、plasma mask、checkpoint selectionは正式契約を維持

実行順序は固定します。

1. 正式SDF
2. 正式Dimension
3. ABC-SDF
4. ABC-Dimension-P

1枚のGPUで順次実行します。既存実測から、純学習は約105～108時間、
前処理、評価、最適化、COMSOL確認、図作成を含めて5～6日を見込みます。

500 epochを50 epoch×10回として実行してはいけません。AdamWの内部状態と
cosine scheduleがリセットされ、連続500 epochとは異なる学習になります。

## 途中再開

25 epochごとに次を保存します。

- model state
- optimizer state
- 現在epoch
- cosine schedule位置
- NumPy乱数状態
- PyTorch CPU/CUDA乱数状態
- 学習履歴
- 現時点のbest validation checkpoint
- config、dataset、実装ファイルのhash

`resume_latest`とbest checkpointだけを保持します。長時間学習前に、
中断・再開した2 stepと連続2 stepの重みが許容誤差内で一致することを
smoke testで確認します。

## 評価

各モデルを次で評価します。

1. 通常test 136件
2. 未知構造75件
3. 未知構造と同一コイル数・同一運転条件の規則anchor 25件

未知構造は各25件の3族です。

- 未知の不等間隔
- 未知の高さ・サイズ順位
- 間隔・高さ・サイズの複合変化

主指標は、物理量へ逆変換した結果を用いた次の値です。

- プラズマ領域の2次元`ni`相対L2誤差
- wafer近傍Bohmフラックス1次元profileの相対L2誤差
- Bohm均一度の絶対誤差
- 未知構造－対応anchorの差分応答誤差

Bohmフラックスは次で定義します。

```text
Gamma_i = ni * sqrt(Te * e / m_Ar)
```

wafer上10層、等面積半径20binを使用します。均一度はprofile平均からの
最大局所偏差率です。全体medianだけでなく90 percentile、worst、
構造族別結果、case対応bootstrapを残します。

## 最適化

最適化は2つに分けます。混合して結論を出してはいけません。

### A：4モデル共通の規則構造最適化

全モデルでコイル数2～6を列挙し、`llcoil`, `rrc`, `rrce`, `zzc`を
同じ範囲、同じcandidate数、同じ局所探索数、optimization seed 4099で
最適化します。`pp=1758.995`, `pp0=0.017931`を固定します。

主目的はwafer Bohmフラックス最大局所偏差の最小化です。低密度にして
均一度だけを改善する解を防ぐため、共通COMSOL anchorから定めた平均密度
下限を使います。

各モデル固有の最良点に加え、4モデルの上位候補の和集合を全モデルで
相互評価します。これによりoptimizerの違いとsurrogate評価の違いを分離します。

### B：SDFの個別コイル構造最適化

SDFモデルでは各コイルの半径位置、高さ、サイズを独立に変えます。
Dimensionモデルは規則構造の最良値を基準として残します。Dimensionモデルが
入力として持たない個別コイル変数を「最適化した」とは表現しません。

上位候補は重複除去後、COMSOLで再計算します。surrogateの順位とCOMSOLの
順位が逆転した場合、最適化成功とはせず、その不一致を結果として報告します。

## 学会用図

最低限使用する図は次の4つです。

1. 図00：構造、COMSOL、4モデルの`ni`空間分布直接比較
2. 図01：同一3ケースのwafer Bohmフラックスprofile
3. 図02：未知75件すべての誤差分布
4. 図07：最適構造とCOMSOL `ni`、Bohm profile

補助図として、構造族別heatmap、未知－anchor差分応答、500 epoch学習曲線、
規則構造最適化、規則対個別コイル設計自由度を作成します。

すべてPNGとSVGで保存し、空間分布は図内で共通の物理color scaleを使います。
主張はlatent距離や複雑な合成指標ではなく、物理分布、Bohm profile、
相対L2、均一度で示します。

## 他環境で最初に行うこと

専用ブランチをcheckoutし、`git lfs pull`でデータ、初期重み、正式参照checkpointを
実体化した後に次を実行します。

```powershell
.\.venv-torch\Scripts\python.exe `
  experiments\icp_stage4\uno_epoch500_four_model_protocol_v47\verify_portability.py `
  --require-formal-references
```

この環境では、凍結source・入力ファイルのSHA256、正式2checkpoint、dataset 957件、
685/136/136 splitがすべて一致し、検証は`pass`しています。

その後の実装・実行は[03_TRAINING_RUNBOOK.md](docs/03_TRAINING_RUNBOOK.md)と
[RUN_CHECKLIST.md](RUN_CHECKLIST.md)に従います。まだ存在しない隔離ランナーを
既存v45/v46ランナーで代用してはいけません。

## 結論判定

ABC-SDFがABC-Dimension-Pより次の両方で良いことが最低条件です。

- 未知75件の`ni` median誤差
- 未知75件のBohm profile median誤差

さらに、3構造族のどれかで重大な悪化がないこと、直接分布が数値順位と
整合すること、入力漏洩がないことを確認します。

最適化の優位性はCOMSOL確認後にのみ主張します。結果が目的を支持しない場合も、
その結果を削除・変更せず、正式モデルを上書きしません。
