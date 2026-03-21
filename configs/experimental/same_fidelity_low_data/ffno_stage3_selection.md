# FFNO Stage 3 Selection

This note defines the operational FFNO baseline for same-fidelity low-data runs.

## Operational baseline
- Keep FFNO on the `B3` settings:
  - `skip_filter: none`
  - `dealias_ratio: 0.85`
  - `taper_alpha: 1.5`
  - `local_skip_cfg.enabled: true`
  - `local_skip_cfg.init_scale: 0.0`
- Use the same target/feature contract as existing FFNO mainline:
  - `target_family: allvars`
  - `target_vars == output_layout.vars`
  - `input_features.mode: geom_feature_pack`
  - `selection.mode: best_val_allvars_balance`

## Experimental-only branches (not promoted)
- Stage 3 C-series:
  - `benchmark_ffno_c1_dual_weight_ne_ni.yaml`
  - `benchmark_ffno_c2_dual_weight_ne_ni_selection.yaml`
  - `benchmark_ffno_c3_dual_weight_ne_ni_selection_init005.yaml`
- Stage 3 D-series:
  - `benchmark_ffno_d1_axis_mix.yaml`

Keep these as ablations only. Do not wire them into mainline compare defaults.

## Result source of truth
- `tmp_ffno/stage3_final_summary.json`

Use this file as the single summary source for Stage 3 decisions.
