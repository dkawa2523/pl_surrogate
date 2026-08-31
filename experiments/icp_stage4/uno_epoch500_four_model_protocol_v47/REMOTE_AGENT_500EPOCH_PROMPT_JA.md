# 他環境エージェントへ渡す500 epoch実行指示

## 使い方

以下のコードブロック全体を、GPUを使用できる他環境のエージェントへ貼り付ける。
この指示は「既存コマンドを実行するだけ」ではなく、未実装のv47隔離ランナー、
ABC-Dimension-P、厳密な途中再開を実装・検証した後、4モデルの500 epoch学習から
評価・最適化・学会図・監査まで完了させる依頼である。

## 貼り付ける指示文

```text
ICP UNOの4モデル500 epoch比較を、実装・学習・評価・最適化・学会用図作成・監査まで完了してください。

対象リポジトリ:
https://github.com/dkawa2523/pl_surrogate.git

使用ブランチ:
docs/icp-uno-epoch500-portable-v47

最低限必要な資産コミット:
0b9d9476a95617743bc8844c03ab39775a70f362
作業開始時は上記ブランチの最新コミットを取得し、この資産コミットを含むことを確認してください。

最初に必ず次を順番に読んでください:
1. experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/REMOTE_TRAINING_START_HERE_JA.md
2. experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/DATASET_ACQUISITION_JA.md
3. experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/docs/01_MODEL_COMPARISON_CONTRACT.md
4. experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/docs/03_TRAINING_RUNBOOK.md
5. experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/docs/04_EVALUATION_AND_OPTIMIZATION.md
6. experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/docs/05_CONFERENCE_FIGURES.md
7. experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/docs/06_RECOVERY_AUDIT_ACCEPTANCE.md
8. experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/RUN_CHECKLIST.md

目的:
次の4モデルを固定データセット、固定split、seed 1237、500 epochで公平に比較してください。

1. Formal SDF
2. Formal Dimension
3. ABC-SDF（適応的局所・大域混合）
4. ABC-Dimension-P（ABC-SDFと同一骨格、同一空間チャネル数、同一学習条件の寸法対照）

重要な現状:
このブランチには957ケースのデータセット、正式モデル設定、初期重み、正式参照checkpoint、ABC-SDF実装、v45/v46参照コード、比較仕様があります。
一方、v47専用run_study.py、smoke_model_contract.py、ABC-Dimension-P、optimizerを含む厳密な途中再開機構は実装対象です。
存在しないv47コマンドを実装済みと仮定せず、最初に隔離実装と検証を行ってください。

取得と初期検証:
- docs/icp-uno-epoch500-portable-v47をcheckoutする。
- git lfs install、git lfs pull、git lfs fsckを実行する。
- DATASET_ACQUISITION_JA.mdどおりにデータ、初期重み、正式参照重みを実体化する。
- Python 3.10環境をrequirements-cuda-cu128-py310.txtから構築し、pip install -e .を実行する。
- GPU、CUDA、空き容量を記録する。
- verify_portability.py --require-formal-referencesを実行する。
- status pass、957 cases、split 685/136/136、warnings 0を確認する。
- 検証失敗時は学習へ進まない。

v47隔離実装:
experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/の下だけに、最低限次を実装してください。

  run_study.py
  smoke_model_contract.py
  configs/generated/

出力先は次だけに限定してください。

  runs/icp_uno_epoch500_four_model_v47/
  reports/icp_conference_materials/icp_uno_epoch500_four_model_v47/

v45/v46のコード、run、report、正式重みを上書きしないでください。

ABC-Dimension-P契約:
- ABC-SDFと同じABC骨格、同じ14 spatial channels、同じ出力headを使う。
- optimizer、epoch、batch size、選択規則をABC-SDFと一致させる。
- trainable parameter数をABC-SDFと一致させる。
- 5寸法broadcast mapを構造群入力とする。
- 5寸法から規則構造を再構成し、その単位電流真空磁場だけを磁場群入力とする。
- 実構造SDFや実不規則構造由来磁場を入力しない。
- ABC-SDF初期重みから意味が一致するtensorだけを移植し、mapping reportを保存する。

500 epoch共通条件:
- epochs: 500
- seed: 1237
- batch size: 16
- optimizer: AdamW
- learning rate: 0.0003
- weight decay: 0.0001
- betas: (0.9, 0.999)
- epsilon: 1e-8
- cosine schedule: 全500 epochで連続
- early stopping: 無効
- validation best checkpoint: 保存
- split: 固定split_membership.json
- 4モデル間でtarget、損失、評価母集団、前処理を恣意的に変更しない。

途中再開:
25 epochごとにmodel、AdamW optimizer、cosine scheduler位置、current epoch、NumPy RNG、PyTorch CPU RNG、全CUDA RNG、history、best-validation state、resolved config hash、dataset hash、model implementation hashを保存してください。
長時間学習前に、別processからresumeした更新が連続実行と許容誤差内で一致するsmoke testを通してください。
500 epochを独立した50 epoch jobの繰返しにしてoptimizerやscheduleを初期化してはいけません。

学習順序:
GPU上では同時実行せず、次の固定順で1モデルずつ実行してください。

1. formal_sdf_500
2. formal_dimension_500
3. abc_sdf_500
4. abc_dimension_p_500

各モデルの開始、100 epoch、250 epoch、500 epoch、評価完了時に、epoch、loss、validation、経過時間、残り時間見積りを報告してください。
中断時は同一hashのcheckpointから再開してください。

完了条件:
各モデルについてRUN_COMPLETE.jsonにstatus complete、epochs_completed 500、selected epoch、elapsed time、resolved config hash、dataset hash、source hash、final checkpoint hash、selected checkpoint hashを記録してください。
ファイルが存在するだけでは完了扱いにしないでください。

500 epoch後の評価:
- standard test 136ケース
- frozen unknown structure 75ケース
- 対応するregular anchor 25ケース
- ni空間分布
- wafer Bohm flux 1D profile
- Bohm均一度
- 未知構造とanchorの差分応答
- 構造族別結果
- paired comparisonとbootstrap uncertainty
- 同一制約・同一candidate条件での最適化
- 学会用PNG/SVG図
- formal資産非改変監査

禁止事項:
- COMSOL教師ケースを学習データへ追加しない。
- COMSOL応答や教師場を入力featureへ入れない。
- split、seed、損失、未知評価母集団を変更しない。
- 結果を良く見せるためcaseや指標を選別しない。
- COMSOL確認前に最適化優位性を物理的事実として主張しない。

最終成果物:
reports/icp_conference_materials/icp_uno_epoch500_four_model_v47/へ次を整理してください。

  TECHNICAL_REPORT.md
  model_summary.csv
  case_metrics_unknown75.csv
  structural_response_metrics_unknown75.csv
  optimization/
  figures/
  CHART_MAP.md
  audit/
  decision_summary.json

最終報告では、4モデルが500 epoch完了した証拠、selected/final epoch、通常test、未知75ケース、構造族別、Bohm flux、最適化、COMSOL確認、学習時間、GPU環境、SDFが良かった点と悪かった点、統計的に断定できない点、学会主張を結果が支持するかを明記してください。
SDF優位性が確認できない場合も結果を変更・削除せず、そのまま結論として報告してください。

GPU不足、ディスク不足、LFS欠損、データhash不一致などで安全に継続できない場合だけ、測定値と不足量を添えて停止してください。それ以外は実装、smoke、resume試験、500 epoch学習、評価、最適化、図、監査まで継続してください。
```

## 関連文書

- データセット取得：`DATASET_ACQUISITION_JA.md`
- 全体入口：`REMOTE_TRAINING_START_HERE_JA.md`
- 500 epoch実行条件：`docs/03_TRAINING_RUNBOOK.md`
- 評価と最適化：`docs/04_EVALUATION_AND_OPTIMIZATION.md`
- 学会用図：`docs/05_CONFERENCE_FIGURES.md`
- 完了・監査条件：`docs/06_RECOVERY_AUDIT_ACCEPTANCE.md`
- 全作業チェック：`RUN_CHECKLIST.md`
