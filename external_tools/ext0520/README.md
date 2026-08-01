# ext0520 Validation Runner (External Tool)

This directory contains orchestration scripts for validating all implemented
models on `outputs_merged_td_all_success_pa_ext0520`-derived datasets.

## Run smoke sweep

```powershell
$env:PYTHONPATH = "src"
.\.venv-torch\Scripts\python external_tools/ext0520/run_ext0520_validation.py --stage smoke --run-compare
```

## Run selected specs only

```powershell
$env:PYTHONPATH = "src"
.\.venv-torch\Scripts\python external_tools/ext0520/run_ext0520_validation.py --stage smoke --only u_no,cno,geom_deeponet_siren
```

## Optional transient reps (benchmark lock note)

`--include-transient` currently uses benchmark entrypoints and may fail if
profile lock requires `axis_mode=steady`.

