# ICP Stage4 Core4 Structure Evaluation

This directory contains the runnable ICP_stage4 evaluation route. The important
constraint is that coil dimensions and positions are not used as scalar training
conditions. For structure runs, scalar conditions are only `pp` and `pp0`; coil
structure is learned through structure feature maps.

For the product-boundary policy, see
`docs/icp_stage4_external_vs_core.md`. In short, ICP-specific objectives,
coil-series constraints, line-profile plots, and publication color scales stay
in external scripts/configs; the core package keeps only generic field-learning
and optimization features.

## Config Policy

Use this README and one curated YAML as the stable entry points. Directories
named `generated*` are reproducible artifacts from
`experiments/icp_stage4/scripts/generate_icp_stage4_core4_benchmark_configs.py`;
do not edit them by hand or treat every generated file as a maintained product
config.

Recommended maintained inputs:

- `conference_structure_feature_study.yaml`
- this README
- the generator scripts used to recreate `generated*` directories

## Dataset

ICP_stage4 is treated as 60 coil-structure groups times 6 process conditions,
for 360 cases. The Core4 targets are `ne`, `ni`, `Te`, and `phi` on a fixed
grid. Splits use `split_group` so cases from the same coil structure stay in
one split.

Primary dataset roots:

- `data/outputs_icp_stage4_enriched_360_csv_npz_core4_struct_spatial_v1`
- `data/outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1`

Reference roots:

- `data/outputs_icp_stage4_enriched_360`
- `data/outputs_icp_stage4_enriched_360_csv_npz_core4_linear`

Before training, check:

- `index.csv` has 360 rows and `index_smoke.csv` has 36 rows.
- `cond_columns` are `pp, pp0` for structure datasets.
- `structure_npz` exists for structure datasets.
- `part_sdf_lite_v1` conversion includes per-case `part_mask_stack` if
  `part_lite_v1` will be evaluated.
- `nncoil` slice counts are reported separately because the fixed test split is
  not perfectly balanced by coil count.

Read-only audit:

```powershell
py -3 scripts\audit_icp_stage4_core4_dataset.py `
  --dataset-root data\outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1 `
  --source-root data\outputs_icp_stage4_enriched_360
```

## Convert

Linear reference dataset:

```powershell
py -3 scripts\convert_outputs_icp_stage4_enriched_to_csv_npz_core4.py `
  --dst-root data\outputs_icp_stage4_enriched_360_csv_npz_core4_linear
```

Structure spatial dataset:

```powershell
py -3 scripts\convert_outputs_icp_stage4_enriched_to_csv_npz_core4.py `
  --structure-spatial-v1 `
  --dst-root data\outputs_icp_stage4_enriched_360_csv_npz_core4_struct_spatial_v1
```

Part dataset for both `icp_part_sdf_lite_v1` and `part_lite_v1`:

```powershell
py -3 scripts\convert_outputs_icp_stage4_enriched_to_csv_npz_core4.py `
  --part-sdf-lite-v1 `
  --dst-root data\outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1
```

The part conversion writes fixed-slot SDF channels and `part_mask_stack`.
`icp_part_sdf_lite_v1` uses the SDF slots; `part_lite_v1` uses order-invariant
part summary channels derived from the stack.

For an existing Core4 dataset created before `part_mask_stack` was persisted,
prepare a corrected dataset without rereading the roughly 50 GB raw COMSOL
exports. Field NPZ files are hard-linked when the filesystem permits it:

```powershell
.venv-torch\Scripts\python.exe scripts\prepare_icp_stage4_part_lite_dataset.py `
  --src-root data\outputs_icp_stage4_enriched_360_csv_npz_core4_part_sdf_lite_v1 `
  --dst-root data\outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2
```

## Generate Configs

Recommended primary run:

```powershell
$env:PYTHONPATH='src;.'
py -3 experiments\icp_stage4\scripts\generate_icp_stage4_core4_benchmark_configs.py `
  --part-lite-v1 `
  --dataset-root data\outputs_icp_stage4_enriched_360_csv_npz_core4_part_lite_v2 `
  --primary-split structure_holdout `
  --models global_mlp deeponet_pod unet unetpp unetpp_attn fno ffno cno cno_operator_unet `
  --sizes smoke full `
  --full-epochs 80 `
  --out-root configs\experimental\icp_stage4\generated_major_fixes_part_lite_v2_e80 `
  --run-root runs\icp_stage4_major_fixes_part_lite_v2_e80
```

Compatibility baselines:

```powershell
py -3 experiments\icp_stage4\scripts\generate_icp_stage4_core4_benchmark_configs.py `
  --part-sdf-lite-v1 `
  --models ffno unet `
  --sizes smoke full `
  --full-epochs 80 `
  --out-root configs\experimental\icp_stage4\generated_part_sdf_lite_v1_e80_primary `
  --run-root runs\icp_stage4_part_sdf_lite_v1_e80_primary

py -3 experiments\icp_stage4\scripts\generate_icp_stage4_core4_benchmark_configs.py `
  --structure-spatial-v1 `
  --models ffno unet cno_operator_unet `
  --sizes smoke full `
  --full-epochs 80 `
  --out-root configs\experimental\icp_stage4\generated_struct_spatial_v1_e80_primary `
  --run-root runs\icp_stage4_struct_spatial_v1_e80_primary
```

For structure configs, the generator writes:

- `dataset.cond_columns: [pp, pp0]`
- `runtime.input_mode: table_plus_structure`
- `runtime.structure.feature_profile` matching the selected profile
- robust condition scaling
- structure-group held-out train/validation/test splits
- required `base_case_id` grouping with fail-fast split validation
- strict case feature-pack and physical `r/z` coordinate contracts
- `log10_floor` for `ne` / `ni`, `log1p` for `Te`,
  `signed_log1p` for `phi`
- train-split-only fitted quantile bounds applied symmetrically during inverse transforms
- plasma-only supervised Huber loss with target weights
- explicit `inference.qoi.uniformity` and positive postprocess defaults

Use `--linear-target-preprocessing` only for legacy comparison.

## Run Training And Evaluation

Smoke first:

```powershell
$env:PYTHONPATH='src;.'
$env:PLASMA_SURROGATE_ENABLE_TORCH='1'
.venv-torch\Scripts\python.exe experiments\icp_stage4\scripts\run_icp_stage4_core4_benchmarks.py `
  --sizes smoke `
  --models ffno deeponet_pod `
  --config-root configs\experimental\icp_stage4\generated_major_fixes_part_lite_v2_e80 `
  --run-root runs\icp_stage4_major_fixes_part_lite_v2_e80 `
  --status-csv runs\icp_stage4_major_fixes_part_lite_v2_e80\smoke_status.csv
```

The 6-group smoke dataset is an integration check, not a selectable scientific
result. Its reliability assessment intentionally fails the minimum 3 held-out
structure-group rule. The full 60-group run resolves to 48/6/6 structure groups
(288/36/36 cases) with no group overlap.

Full comparison:

```powershell
.venv-torch\Scripts\python.exe experiments\icp_stage4\scripts\run_icp_stage4_core4_benchmarks.py `
  --sizes full `
  --models ffno deeponet_pod unet cno_operator_unet `
  --config-root configs\experimental\icp_stage4\generated_major_fixes_part_lite_v2_e80 `
  --run-root runs\icp_stage4_major_fixes_part_lite_v2_e80 `
  --status-csv runs\icp_stage4_major_fixes_part_lite_v2_e80\run_status.csv

.venv-torch\Scripts\python.exe experiments\icp_stage4\scripts\summarize_icp_stage4_core4_benchmarks.py `
  --sizes full `
  --models ffno deeponet_pod unet cno_operator_unet `
  --config-root configs\experimental\icp_stage4\generated_major_fixes_part_lite_v2_e80 `
  --run-root runs\icp_stage4_major_fixes_part_lite_v2_e80 `
  --out-dir runs\icp_stage4_major_fixes_part_lite_v2_e80\summary
```

For the latest selected UNO recipe, run independent seeds and rank only by the
validation mean/standard deviation. Test metrics remain reporting-only:

```powershell
.venv-torch\Scripts\python.exe experiments\icp_stage4\scripts\run_icp_stage4_uno_tuning_plan.py `
  --config-root configs\experimental\icp_stage4\generated_major_fixes_part_lite_v2_e80 `
  --uno-config-root configs\experimental\icp_stage4\generated_major_fixes_part_lite_v2_e80_uno3 `
  --run-root runs\icp_stage4_major_fixes_part_lite_v2_e80_uno3 `
  --epochs 70 `
  --seeds 411 412 413 `
  --only-recipe u_no_m10_w48_l4_lr3e4 `
  --stop-on-failure
```

Choose optimization surrogates by both field metrics and QoI metrics. FFNO,
UNet, and CNO/operator runs should all be treated as production candidates; the
best choice depends on the target field group and distribution-quality metrics.

## Validate Spatial Learning Before Optimization

Generate the three controlled U-Net conditions, or run them sequentially:

```powershell
.venv-torch\Scripts\python.exe experiments\icp_stage4\scripts\run_icp_stage4_spatial_learning_plan.py

.venv-torch\Scripts\python.exe experiments\icp_stage4\scripts\run_icp_stage4_spatial_learning_plan.py `
  --run `
  --variants transform_case_balance smooth_structure smooth_spatial_loss
```

The study fixes seed 411, the structure-holdout split, U-Net compact, and 70
epochs. It changes only target clipping/case aggregation, smooth structure
features, and spatial loss in successive conditions. Runs and generated configs
are isolated under `icp_stage4_spatial_learning_v1`. Do not resume shape
optimization until the case-level physical, gradient, multiscale, and bound
saturation diagnostics pass.

## Optimize Coil Structure And Process

Use the trained structure surrogate and parametric parts provider. Do not
optimize raw masks or feed raw coil parameter vectors to the model.

Process + layout smoke:

```powershell
.venv-torch\Scripts\python.exe experiments\icp_stage4\scripts\run_icp_part_sdf_shape_optimize.py `
  --config configs\experimental\icp_stage4\generated_part_lite_v1_e80_primary\full\benchmark_icp_stage4_core4_full_ffno.yaml `
  --run-dir runs\icp_stage4_part_lite_v1_e80_primary\full\ffno `
  --model ffno `
  --out-dir reports\icp_stage4_part_lite_v1_optimize_smoke\ffno `
  --reference-case-id case_g002_op01 `
  --space-mode layout `
  --condition-mode dataset_bounds `
  --n-trials 32 `
  --n-initial 16 `
  --top-k 4 `
  --local-trials-per-seed 4
```

Final candidate search:

```powershell
.venv-torch\Scripts\python.exe experiments\icp_stage4\scripts\run_icp_part_sdf_shape_optimize.py `
  --config configs\experimental\icp_stage4\generated_part_lite_v1_e80_primary\full\benchmark_icp_stage4_core4_full_ffno.yaml `
  --run-dir runs\icp_stage4_part_lite_v1_e80_primary\full\ffno `
  --model ffno `
  --out-dir reports\icp_stage4_part_lite_v1_optimize_final\ffno `
  --reference-case-id case_g002_op01 `
  --space-mode layout `
  --condition-mode dataset_bounds `
  --n-trials 512
```

The optimizer uses `two_stage` by default with a normalized `weighted_sum`
objective. Feasible trials are selected first, and physics / boundary terms are
configured as ordinary objective terms or constraints. Re-score the top FFNO
candidates with UNet/CNO before selecting 3-5 designs for high-fidelity COMSOL
validation.
