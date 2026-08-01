# Structure Converter (External Tool)

This directory contains dataset conversion utilities that are intentionally
kept outside `src/plasma_surrogate`.

## mphtxt -> parametric_parts

After running `scripts/convert_outputs_merged_td_to_csv_npz.py`, generate
provider-compatible v2 geometry artifacts:

- `geometry/parts_manifest.json`
- `geometry/parts_pack.npz`
- `structure_features/{base2,base3,base4}.npz`

The converter reads the MPHTXT `edg` connectivity and geometric entity IDs
directly. It rasterizes the common 12 entity IDs into fixed semantic slots for
each mutually exclusive chamber alternative. The manifest therefore records
`part_semantics=alternatives`, `plasma_mode=preserve`, and
`approximate=false`.

Command example:

```powershell
.\.venv-torch\Scripts\python.exe external_tools/structure_converter/convert_mphtxt_to_parametric_parts.py
```

The command defaults to source
`data/outputs_merged_td_all_success_pa_ext0520` and destination
`data/outputs_merged_td_csv_periodic_ext0520_v2`. Explicit path flags remain
available for other datasets.
