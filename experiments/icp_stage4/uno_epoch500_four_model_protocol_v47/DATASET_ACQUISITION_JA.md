# 固定データセットとGit LFS資産の取得方法

## 対象

500 epoch比較で使用するデータセットは、次の957ケースに固定する。

```text
data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/
```

splitも次に固定し、再生成しない。

```text
train       685
validation  136
test        136
```

配列、初期重み、正式参照checkpointはGit LFSで管理されている。通常の
`git clone`だけで取得したファイルがLFS pointerのままなら、学習を開始できない。

## 推奨取得方法：全LFS資産を取得

Windows PowerShell:

```powershell
git clone --branch docs/icp-uno-epoch500-portable-v47 --single-branch `
  https://github.com/dkawa2523/pl_surrogate.git
Set-Location pl_surrogate
git lfs install
git lfs pull
git lfs fsck
```

Linux:

```bash
git clone --branch docs/icp-uno-epoch500-portable-v47 --single-branch \
  https://github.com/dkawa2523/pl_surrogate.git
cd pl_surrogate
git lfs install
git lfs pull
git lfs fsck
```

この方法は、固定データセット、3種類の初期重み、正式2モデルの参照重みを
まとめて取得する。十分な空き容量がある場合は、この方法を使用する。

## 容量を限定する場合

LFS対象を限定する場合も、データだけでなく初期重みと正式参照重みを含める。

PowerShell:

```powershell
$include = @(
  'data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/**',
  'configs/experimental/icp_stage4/conference_continuity_v45/initial_weights/*.npz',
  'experiments/icp_stage4/uno_structure_em_v46/configs/initial_weights/*.npz',
  'runs/icp_conference_continuity_v45/final/conference_sdf/seed_1237/models/u_no/eval_protocol/interp/checkpoints/weights.npz',
  'runs/icp_conference_continuity_v45/final/conference_dimension/seed_1237/models/u_no/eval_protocol/interp/checkpoints/weights.npz'
) -join ','
git lfs pull --include=$include
git lfs fsck
```

Linux:

```bash
git lfs pull --include='data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/**,configs/experimental/icp_stage4/conference_continuity_v45/initial_weights/*.npz,experiments/icp_stage4/uno_structure_em_v46/configs/initial_weights/*.npz,runs/icp_conference_continuity_v45/final/conference_sdf/seed_1237/models/u_no/eval_protocol/interp/checkpoints/weights.npz,runs/icp_conference_continuity_v45/final/conference_dimension/seed_1237/models/u_no/eval_protocol/interp/checkpoints/weights.npz'
git lfs fsck
```

## 学習前の必須検証

Python環境を構築した後、リポジトリrootから次を実行する。

Windows PowerShell:

```powershell
.\.venv-torch\Scripts\python.exe `
  experiments\icp_stage4\uno_epoch500_four_model_protocol_v47\verify_portability.py `
  --require-formal-references
```

Linux:

```bash
.venv-torch/bin/python \
  experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/verify_portability.py \
  --require-formal-references
```

合格条件は次のとおり。

- `status`が`pass`
- `required_file_count_checked`が24
- `dataset_case_count`が957
- splitが685 / 136 / 136
- `warnings`と`failures`が空
- `git lfs fsck`が成功

不合格なら学習へ進まず、branch、LFS取得、ファイル欠損、hash不一致を解消する。
別データセットへの差し替え、split再生成、欠損ケースの除外は禁止する。

## 対応する仕様書

- 入力資産と取得全体：`REMOTE_TRAINING_START_HERE_JA.md`
- ファイルhash：`SOURCE_INVENTORY.json`
- データ・比較契約：`docs/01_MODEL_COMPARISON_CONTRACT.md`
- 環境移行：`docs/02_ENVIRONMENT_AND_TRANSFER.md`
- 500 epoch実行：`docs/03_TRAINING_RUNBOOK.md`
