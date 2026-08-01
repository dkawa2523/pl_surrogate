# Benchmark Runbook

This runbook reproduces 27 / 54 / 78 case-size benchmark runs with the
`outputs_merged_td_all_success_pa_ext0520` dataset.  Commands are written for
Windows PowerShell from the repository root.

## 1. Environment

```powershell
Set-Location C:\Users\user\Desktop\pl_surrogate
py -3.10 -m venv .venv-torch
.\.venv-torch\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-torch\Scripts\python.exe -m pip install -r requirements-cuda-cu128-py310.txt
.\.venv-torch\Scripts\python.exe -m pip install -e ".[dev]"
$env:PLASMA_SURROGATE_ENABLE_TORCH = "1"
```

Use the installed CLI:

```powershell
$cli = ".\.venv-torch\Scripts\plasma-surrogate.exe"
```

## 2. Dataset

The raw ext0520 output is expected at:

```text
data/outputs_merged_td_all_success_pa_ext0520
```

Check the required files:

```powershell
Test-Path data/outputs_merged_td_all_success_pa_ext0520/export_file_index.csv
Test-Path data/outputs_merged_td_all_success_pa_ext0520/conditions.csv
```

Convert to the package `csv_npz` dataset format if needed:

```powershell
.\.venv-torch\Scripts\python.exe scripts/convert_outputs_merged_td_to_csv_npz.py `
  --src-root data/outputs_merged_td_all_success_pa_ext0520 `
  --dst-root data/outputs_merged_td_csv_periodic_ext0520 `
  --mode periodic `
  --cond-columns PP0,Td,gamma
```

## 3. Create 27 / 54 / 78 Index Files

The split below is deterministic: cases are ordered by SHA1 of `case_id`.

```powershell
@'
import csv
import hashlib
from pathlib import Path

root = Path("data/outputs_merged_td_csv_periodic_ext0520")
src = root / "index.csv"
rows = list(csv.DictReader(src.open("r", encoding="utf-8")))
rows = sorted(rows, key=lambda r: hashlib.sha1(r["case_id"].encode("utf-8")).hexdigest())

for n in (27, 54, 78):
    out = root / f"index_{n}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows[:n])
    print(out, "rows=", n)
'@ | .\.venv-torch\Scripts\python.exe -
```

Expected line counts, including the header:

```powershell
(Get-Content data/outputs_merged_td_csv_periodic_ext0520/index_27.csv | Measure-Object -Line).Lines
(Get-Content data/outputs_merged_td_csv_periodic_ext0520/index_54.csv | Measure-Object -Line).Lines
(Get-Content data/outputs_merged_td_csv_periodic_ext0520/index_78.csv | Measure-Object -Line).Lines
```

Expected values: `28`, `55`, `79`.

## 4. Generate Benchmark Configs

This runbook keeps every supported comparison model in the run.  Curated
templates live under `configs/benchmarkrun_ext0520/templates`.  Generated
per-size configs are local artifacts under `runs/benchmarkrun_ext0520/configs`;
do not place them under `tests/fixtures/generated`.

Model set:

- `global_mlp`
- `deeponet_pod`
- `unet`
- `unetpp`
- `unetpp_attn`
- `fno`
- `ffno`
- `coord_mlp_fourier`
- `coord_mlp_siren`
- `coord_mlp_pod_residual`
- `u_no`
- `cno`
- `cno_operator_unet`
- `geom_deeponet_pod`
- `geom_deeponet_siren`
- `deeponet_plasma_pod`
- `deeponet_plasma`

Generate per-size configs:

```powershell
.\.venv-torch\Scripts\python.exe scripts/generate_benchmarkrun_ext0520_configs.py --sizes 27 54 78
```

## 5. Run Benchmarks

```powershell
$env:PLASMA_SURROGATE_ENABLE_TORCH = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

.\.venv-torch\Scripts\python.exe scripts/run_benchmarkrun_ext0520.py --sizes 78 --stop-on-failure
```

The runner executes one model at a time, lowers the Python process priority on
Windows, limits common CPU thread pools, skips existing `leaderboard.csv` files,
and updates `runs/benchmarkrun_ext0520/run_status.csv`.  To re-check already
completed sizes without retraining them, pass `--sizes 27 54 78`.

## 6. Summarize Results

Each run writes:

- `runs/benchmarkrun_ext0520/n{27|54|78}/{model}/leaderboard.csv`
- `runs/benchmarkrun_ext0520/n{27|54|78}/{model}/resolved_benchmark.json`

Collect the primary columns:

```powershell
@'
import csv
import json
from pathlib import Path

models = [
    "global_mlp",
    "deeponet_pod",
    "unet",
    "unetpp",
    "unetpp_attn",
    "fno",
    "ffno",
    "coord_mlp_fourier",
    "coord_mlp_siren",
    "coord_mlp_pod_residual",
    "u_no",
    "cno",
    "cno_operator_unet",
    "geom_deeponet_pod",
    "geom_deeponet_siren",
    "deeponet_plasma_pod",
    "deeponet_plasma",
]

rows = []
for n in (27, 54, 78):
    for model in models:
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
            "primary_metric": top.get("primary_metric", ""),
            "primary_metric_value": top.get("primary_metric_value", ""),
            "test_r2_plasma_mean_dual": top.get("test_r2_plasma_mean_dual", ""),
            "test_r2_plasma_mean_interp": top.get("test_r2_plasma_mean_interp", ""),
            "test_r2_plasma_mean_extrap": top.get("test_r2_plasma_mean_extrap", ""),
            "sdf_boundary_to_deep_rmse_ratio_mean": top.get("sdf_boundary_to_deep_rmse_ratio_mean", ""),
            "sdf_boundary_minus_deep_r2_mean": top.get("sdf_boundary_minus_deep_r2_mean", ""),
            "leaderboard_csv": str(lb),
        })

out = Path("runs/benchmarkrun_ext0520/summary_27_54_78_all_models.csv")
out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
print(out)
'@ | .\.venv-torch\Scripts\python.exe -
```

SDF-derived columns help identify whether a model degrades near the plasma
boundary compared with deep plasma:

- `sdf_boundary_to_deep_rmse_ratio_mean`: values above `1` mean boundary error
  is larger than deep-plasma error.
- `sdf_boundary_minus_deep_r2_mean`: negative values mean boundary R2 is worse
  than deep-plasma R2.

Keep committed test fixtures small.  When a benchmark recipe becomes a formal
experiment template, move the template material to `configs/`; keep completed
run outputs, logs, and generated per-size configs under `runs/`.
