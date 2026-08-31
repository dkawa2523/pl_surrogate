# 他環境での500 epoch学習：最初に読む文書

## 最短の参照順序

1. 他環境エージェントへ渡す指示：
   [REMOTE_AGENT_500EPOCH_PROMPT_JA.md](REMOTE_AGENT_500EPOCH_PROMPT_JA.md)
2. データセットとLFS資産の取得：
   [DATASET_ACQUISITION_JA.md](DATASET_ACQUISITION_JA.md)
3. モデル間の公平比較契約：
   [docs/01_MODEL_COMPARISON_CONTRACT.md](docs/01_MODEL_COMPARISON_CONTRACT.md)
4. 500 epochの実装・学習・resume：
   [docs/03_TRAINING_RUNBOOK.md](docs/03_TRAINING_RUNBOOK.md)
5. 学習後の評価・最適化・図・監査：
   [docs/04_EVALUATION_AND_OPTIMIZATION.md](docs/04_EVALUATION_AND_OPTIMIZATION.md)、
   [docs/05_CONFERENCE_FIGURES.md](docs/05_CONFERENCE_FIGURES.md)、
   [docs/06_RECOVERY_AUDIT_ACCEPTANCE.md](docs/06_RECOVERY_AUDIT_ACCEPTANCE.md)

エージェントへ依頼するときは、1のMarkdownをそのまま貼り付ける。
データ取得だけを別担当者が行う場合は2を渡す。

## このブランチで受け取れるもの

`docs/icp-uno-epoch500-portable-v47` ブランチは、他環境でv47を実装・学習するための
凍結入力を一か所にまとめる。

- 957件の固定データセット（Git LFS）
- 現在の `src/plasma_surrogate` Python実装snapshot
- 正式SDF／正式Dimensionの採用設定と初期重み
- ABC-SDFのv46隔離モデル、factory hook、設定、初期重み
- 正式2モデルの参照checkpoint
- 4モデル500 epochの比較・評価・最適化・図・監査仕様

このブランチは500 epochの結果を含まない。また、v47専用の
`run_study.py`、`smoke_model_contract.py`、ABC-Dimension-P、およびoptimizerを含む
途中再開機構は実装対象である。既存v45/v46のコードを実装例として読み、
v45/v46のrun rootへ書き込まず、v47専用領域へ実装する。

## 取得

詳細、容量を限定した取得方法、検証条件は
[DATASET_ACQUISITION_JA.md](DATASET_ACQUISITION_JA.md)を正とする。

```powershell
git clone --branch docs/icp-uno-epoch500-portable-v47 --single-branch `
  https://github.com/dkawa2523/pl_surrogate.git
Set-Location pl_surrogate
git lfs install
git lfs pull
git lfs fsck
```

LFSを特定領域だけ取得する場合は、少なくともデータ、初期重み、参照checkpointを
含める。単純化のため、十分な空き容量があれば最初は `git lfs pull` を推奨する。

## Python環境

```powershell
py -3.10 -m venv .venv-torch
.\.venv-torch\Scripts\python.exe -m pip install --upgrade pip
.\.venv-torch\Scripts\python.exe -m pip install -r requirements-cuda-cu128-py310.txt
.\.venv-torch\Scripts\python.exe -m pip install -e .
$env:PLASMA_SURROGATE_ENABLE_TORCH='1'
$env:PYTHONPATH=(Resolve-Path -LiteralPath 'src').Path
```

## 最初の合否判定

```powershell
nvidia-smi
.\.venv-torch\Scripts\python.exe `
  experiments\icp_stage4\uno_epoch500_four_model_protocol_v47\verify_portability.py `
  --require-formal-references
```

期待値は `status: pass`、dataset 957件、split 685/136/136、警告0である。
失敗時は学習へ進まず、欠損パスまたはSHA256不一致を解消する。

## 実装元の対応表

| v47モデル | 採用設定／実装元 | 初期重み |
|---|---|---|
| Formal SDF | `configs/.../conference_continuity_v45/final/conference_sdf/seed_1237.yaml` と現行CLI | `conference_sdf_seed1237.npz` |
| Formal Dimension | `configs/.../conference_continuity_v45/final/conference_dimension/seed_1237.yaml` と現行CLI | `conference_dimension_seed1237.npz` |
| ABC-SDF | `experiments/icp_stage4/uno_structure_em_v46/experimental_uno.py`、`run_entry.py`、ABC設定 | `ABC_adaptive_mix_seed1237.npz` |
| ABC-Dimension-P | `docs/01_MODEL_COMPARISON_CONTRACT.md`に従い新規隔離実装 | ABC-SDFから意味の一致するtensorだけを移植しmappingを保存 |

Formal pairの生成・評価例は
`experiments/icp_stage4/conference_continuity_v45/`、ABC-SDFの隔離factory、
warm start、smoke、評価例は `experiments/icp_stage4/uno_structure_em_v46/` にある。
これらは参照専用で、v47の出力先として使わない。

## v47で実装するもの

専用ディレクトリ内に最低限次を作る。

```text
experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/
  run_study.py
  smoke_model_contract.py
  configs/generated/
```

出力は次だけに限定する。

```text
runs/icp_uno_epoch500_four_model_v47/
reports/icp_conference_materials/icp_uno_epoch500_four_model_v47/
```

ABC-Dimension-Pは実構造を参照せず、5寸法broadcast mapと、5寸法から再構成した
規則構造の単位電流真空磁場だけを入力する。ABC-SDFと同じ14 spatial channels、
同じABC骨格、出力head、optimizer、epoch数、trainable parameter数にする。

途中再開は25 epochごとにmodel、AdamW、cosine位置、epoch、NumPy／PyTorch／CUDA
乱数、history、best validation、config・dataset・実装hashを保存する。
500 epochを独立した短いjobへ分割してoptimizerを初期化してはいけない。

## 実行順と完了条件

実行順は Formal SDF、Formal Dimension、ABC-SDF、ABC-Dimension-P の固定順で、
GPU上で同時実行しない。詳細は `docs/03_TRAINING_RUNBOOK.md` に従う。

各モデルは `RUN_COMPLETE.json` が `epochs_completed: 500` と
`status: complete` を示し、checkpoint・metrics・resolved configのhashが保存されて
初めて完了である。次にevaluate、optimize、figures、auditを実行する。

## 禁止事項

- v45/v46のコード、重み、run、reportを上書きしない。
- COMSOL応答や教師場を入力featureへ入れない。
- split、seed、損失、評価母集団を変更しない。
- ABC-Dimension-Pへ実不規則構造由来の磁場を入れない。
- 結果が主張を支持しない場合にcaseや指標を選別しない。

作業全体のチェックは `RUN_CHECKLIST.md`、判定規則は
`docs/06_RECOVERY_AUDIT_ACCEPTANCE.md` を正とする。
