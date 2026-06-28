# Structure Converter (External Tool)

This directory contains dataset conversion utilities that are intentionally
kept outside `src/plasma_surrogate`.

## mphtxt -> parametric_parts

Generate provider-compatible geometry artifacts:

- `geometry/parts_manifest.json`
- `geometry/parts_pack.npz`

Command example:

```powershell
$env:PYTHONPATH = "src"
.\.venv-torch\Scripts\python external_tools/structure_converter/convert_mphtxt_to_parametric_parts.py `
  --source-root data/outputs_merged_td_all_success_pa_ext0520 `
  --structure-index-csv data/outputs_merged_td_all_success_pa_ext0520/structure_file_index.csv `
  --conditions-with-structure-csv data/outputs_merged_td_all_success_pa_ext0520/conditions_with_structure.csv `
  --dataset-root data/outputs_merged_td_csv_periodic_ext0520
```

