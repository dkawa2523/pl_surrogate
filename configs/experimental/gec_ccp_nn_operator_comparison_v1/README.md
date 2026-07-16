# GEC-CCP NN / Neural-Operator Comparison v1

> Historical baseline: this directory retains the original POD coefficient
> branch.  The validation-selected branch with GELU after every hidden layer
> is adopted in `../gec_ccp_nn_operator_comparison_v2/`.  Do not overwrite the
> v1 run artifacts when reproducing v2.

This suite compares eight representative neural architectures under one
spatial-learning and validation protocol.  It is intentionally narrower than
the complete model registry.

## Common comparison contract

- dataset: `outputs_merged_td_csv_periodic_ext0520_v2/index_78.csv`
- condition vector: `PP0, Td, gamma, PA`
- targets: `ne, ni, Te, phi` in linear physical values
- target preprocessing: identity transform, train-only plasma-region z-score
- fixed split seed: `7`; interpolation train/validation/test = `54/11/13`
- evaluation lane: `primary_axis: interp` with `interp_mode: marginal`
- loss: `plasma_surrogate_v3` Huber + gradient + multiscale + boundary terms
- checkpoint: validation-only `best_val_spatial_objective`
- learning seeds: `411, 412, 413`

The PP0 stress lane is excluded from this primary comparison because it trains
only on `PP0=1`.  It can be run later as a separately labelled extrapolation
diagnostic; it must not be mixed with the 54/11/13 interpolation result.

## Models

| Group | Model | Main model-specific setting |
| --- | --- | --- |
| vector NN | `global_mlp` | current `[128, 128]` reference |
| vector NN | `global_resmlp` | width 48, 3 residual blocks |
| vector NN | `global_densemlp` | width 48, 3 dense blocks, growth 12 |
| convolutional NN | `unet` | base width 48, depth 2, resize-convolution upsampling |
| neural operator | `fno` | current width 96, 12 modes, 4 layers reference |
| neural operator | `ffno` | width 96, 16 factorized modes, 4 layers |
| neural operator | `u_no` | width 96, 12 modes, 4 layers |
| POD operator | `deeponet_pod` | historical 192x2 branch, GELU only between hidden layers, rank cap 24 |

`global_resmlp` and `global_densemlp` use width 48 because each prediction has
`4 x 204 x 201 = 164,016` outputs and only 54 training cases.  This keeps the
full-field head near 7.9 million weights and the ridge design at 49 columns.
These are validation candidates, not settings already tuned on GEC-CCP.

## Input-contract interpretation

The dataset, targets, loss, split, and metrics are identical, but architecture
input representations are necessarily different:

- `global_*`: condition table only; geometry is used only for loss/selection.
- `unet`, `fno`, `ffno`, `u_no`: condition table plus the same 13-channel
  `part_lite_v1` spatial pack.
- `deeponet_pod`: condition table plus `struct_desc_v2`.

In these 78 cases, `Td` and the three structures are one-to-one associated.
The structure pack therefore adds spatial inductive bias, not an independently
crossed physical variable.  Do not interpret this suite as unknown-geometry
generalization.

## Regenerate

```powershell
$env:PLASMA_SURROGATE_ENABLE_TORCH = "1"
.\.venv-torch\Scripts\python.exe `
  scripts/prepare_gec_ccp_nn_operator_comparison_v1_configs.py
```

## Run one seed

```powershell
.\.venv-torch\Scripts\python.exe scripts/run_benchmarkrun_ext0520.py `
  --sizes 78 `
  --models global_mlp global_resmlp global_densemlp unet fno ffno u_no deeponet_pod `
  --config-root configs/experimental/gec_ccp_nn_operator_comparison_v1/seed_412 `
  --run-root runs/gec_ccp_nn_operator_comparison_v1/seed_412 `
  --status-csv runs/gec_ccp_nn_operator_comparison_v1/seed_412/run_status.csv `
  --load-profile normal `
  --stop-on-failure
```

Use the same command with seeds `411` and `413` for the final mean/std.  Model
or recipe selection must use validation values only; inspect test metrics only
after the three-seed matrix is frozen.
