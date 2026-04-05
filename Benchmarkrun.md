# Benchmarkrun.md

`outputs_merged_td_all_success_pa_ext0520` を使って、`27 / 54 / 78` ケースで学習・評価を回すための手順です。  
初めて触る人がそのまま実行できるように、PowerShell 前提でコマンドを全部書いています。

## 1. 前提

- OS: Windows + PowerShell
- Python: 3.10
- 実行ディレクトリ: リポジトリルート（このファイルがある場所）

```powershell
Set-Location c:\Users\user\Desktop\pl_surrogate
```

## 2. 環境構築

### 2.1 仮想環境作成

```powershell
py -3.10 -m venv .venv-torch
```

### 2.2 依存インストール（CUDA版）

```powershell
.\.venv-torch\Scripts\python.exe -m pip install --upgrade pip
.\.venv-torch\Scripts\python.exe -m pip install -r requirements-cuda-cu128-py310.txt
```

### 2.3 実行時環境変数

```powershell
$env:PYTHONPATH = "src"
$env:PLASMA_SURROGATE_ENABLE_TORCH = "1"
```

## 3. データ準備

このリポジトリでは、完全版ソースを `data/outputs_merged_td_all_success_pa_ext0520` に置く想定です。  
（`export_file_index.csv` が必要）

### 3.1 ソースデータ確認

```powershell
Test-Path data/outputs_merged_td_all_success_pa_ext0520/export_file_index.csv
Test-Path data/outputs_merged_td_all_success_pa_ext0520/conditions.csv
```

どちらかが `False` の場合は、データ配置を見直してください。

### 3.2 `csv_npz` 形式へ変換（未作成の場合のみ）

既に `data/outputs_merged_td_csv_periodic_ext0520/index.csv` があるならスキップ可です。

```powershell
.\.venv-torch\Scripts\python.exe scripts/convert_outputs_merged_td_to_csv_npz.py `
  --src-root data/outputs_merged_td_all_success_pa_ext0520 `
  --dst-root data/outputs_merged_td_csv_periodic_ext0520 `
  --mode periodic `
  --cond-columns PP0,Td,gamma
```

## 4. 27/54/78 用 index を作る

`index.csv`（78ケース）から、再現可能な方法で `index_27.csv` / `index_54.csv` / `index_78.csv` を作ります。  
（`case_id` の SHA1 順で切り出し）

```powershell
@'
import csv, hashlib
from pathlib import Path

root = Path("data/outputs_merged_td_csv_periodic_ext0520")
src = root / "index.csv"
rows = list(csv.DictReader(src.open("r", encoding="utf-8")))
rows = sorted(rows, key=lambda r: hashlib.sha1(r["case_id"].encode("utf-8")).hexdigest())

for n in (27, 54, 78):
    out = root / f"index_{n}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows[:n])
    print(out, "rows=", n)
'@ | .\.venv-torch\Scripts\python.exe -
```

作成確認:

```powershell
(Get-Content data/outputs_merged_td_csv_periodic_ext0520/index_27.csv | Measure-Object -Line).Lines
(Get-Content data/outputs_merged_td_csv_periodic_ext0520/index_54.csv | Measure-Object -Line).Lines
(Get-Content data/outputs_merged_td_csv_periodic_ext0520/index_78.csv | Measure-Object -Line).Lines
```

期待値は `28 / 55 / 79`（ヘッダ込み）です。

## 5. 3サイズ用 benchmark fixture を自動生成

ここでは mainline 4モデルを対象にします。

- `global_mlp`（frozen ref）
- `unet`
- `fno`
- `deeponet_plasma`

```powershell
@'
from pathlib import Path
import yaml

base = {
    "global_mlp": "tests/fixtures/benchmark_periodic_real_m7_global_frozen_ref.yaml",
    "unet": "tests/fixtures/benchmark_periodic_real_m7_unet_isolated_mainline.yaml",
    "fno": "tests/fixtures/benchmark_periodic_real_m7_fno_isolated_mainline.yaml",
    "deeponet_plasma": "tests/fixtures/benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml",
}

for n in (27, 54, 78):
    for model, src in base.items():
        cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
        b = cfg["benchmark"]
        b["dataset"]["root"] = "data/outputs_merged_td_csv_periodic_ext0520"
        b["dataset"]["index_csv"] = f"index_{n}.csv"
        b["output_dir"] = f"runs/benchmarkrun_ext0520/n{n}/{model}"
        out = Path(f"tests/fixtures/generated/benchmarkrun_ext0520/n{n}/benchmark_ext0520_{model}_n{n}.yaml")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(out)
'@ | .\.venv-torch\Scripts\python.exe -
```

## 6. 学習 + 評価（benchmark run）

`benchmark run` は、前処理・学習・推論・評価・leaderboard 出力までまとめて実行します。

```powershell
$py = ".\.venv-torch\Scripts\python.exe"
$sizes = @(27, 54, 78)
$models = @("global_mlp", "unet", "fno", "deeponet_plasma")

foreach ($n in $sizes) {
  foreach ($m in $models) {
    $cfg = "tests/fixtures/generated/benchmarkrun_ext0520/n$($n)/benchmark_ext0520_$($m)_n$($n).yaml"
    Write-Host "[RUN] n=$n model=$m"
    & $py -m plasma_surrogate.cli.main benchmark run --config $cfg
    if ($LASTEXITCODE -ne 0) { throw "failed: $cfg" }
  }
}
```

## 7. 結果確認

### 7.1 各 run の主要成果物

- `runs/benchmarkrun_ext0520/n{27|54|78}/{model}/leaderboard.csv`
- `runs/benchmarkrun_ext0520/n{27|54|78}/{model}/resolved_benchmark.json`

### 7.2 主要指標を1枚のCSVにまとめる

```powershell
@'
import csv, json
from pathlib import Path

rows = []
for n in (27, 54, 78):
    for model in ("global_mlp", "unet", "fno", "deeponet_plasma"):
        run = Path(f"runs/benchmarkrun_ext0520/n{n}/{model}")
        lb = run / "leaderboard.csv"
        rb = run / "resolved_benchmark.json"
        if not lb.exists() or not rb.exists():
            continue
        top = next(csv.DictReader(lb.open("r", encoding="utf-8")))
        resolved = json.loads(rb.read_text(encoding="utf-8"))
        rows.append({
            "dataset_size": n,
            "model_id": model,
            "n_cases_effective": resolved["dataset"]["n_cases"],
            "test_r2_plasma_mean_dual": top.get("test_r2_plasma_mean_dual", ""),
            "test_r2_plasma_mean_interp": top.get("test_r2_plasma_mean_interp", ""),
            "test_r2_plasma_mean_extrap": top.get("test_r2_plasma_mean_extrap", ""),
            "leaderboard_csv": str(lb),
        })

out = Path("runs/benchmarkrun_ext0520/summary_27_54_78_mainline.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(out)
'@ | .\.venv-torch\Scripts\python.exe -
```

## 8. compare（任意）

`global_mlp` は `table_only` なので、`table_plus_structure` の3モデル比較は分けて実施するのが安全です。

```powershell
@'
from pathlib import Path
import yaml

for n in (27, 54, 78):
    cfg = {
      "compare": {
        "output_dir": f"runs/benchmarkrun_ext0520/n{n}/compare_unet_fno_deeponet",
        "require_same_input_mode": True,
        "global_reference_mode": "off",
        "objective_metric": "test_r2_plasma_mean_dual",
        "objective_mode": "max",
        "rows": [
          {"name": "unet", "model_id": "unet", "leaderboard_csv": f"runs/benchmarkrun_ext0520/n{n}/unet/leaderboard.csv"},
          {"name": "fno", "model_id": "fno", "leaderboard_csv": f"runs/benchmarkrun_ext0520/n{n}/fno/leaderboard.csv"},
          {"name": "deeponet_plasma", "model_id": "deeponet_plasma", "leaderboard_csv": f"runs/benchmarkrun_ext0520/n{n}/deeponet_plasma/leaderboard.csv"},
        ]
      }
    }
    out = Path(f"tests/fixtures/generated/benchmarkrun_ext0520/n{n}/compare_unet_fno_deeponet_n{n}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(out)
'@ | .\.venv-torch\Scripts\python.exe -
```

```powershell
$py = ".\.venv-torch\Scripts\python.exe"
foreach ($n in 27,54,78) {
  & $py scripts/compare_selected_models.py --config "tests/fixtures/generated/benchmarkrun_ext0520/n$($n)/compare_unet_fno_deeponet_n$($n).yaml"
}
```

## 9. よくある詰まりどころ

- `export_file_index.csv` が見つからない  
  `data/outputs_merged_td_all_success_pa_ext0520` の完全版を使ってください。
- `PLASMA_SURROGATE_ENABLE_TORCH` 未設定  
  torchモデルの benchmark で失敗します。`$env:PLASMA_SURROGATE_ENABLE_TORCH="1"` を設定してください。
- `interp_mode=overlap` なのに fallback warning が出る  
  subset の取り方次第で発生します。`resolved_benchmark.json` の `interp_mode_effective` を確認してください。

