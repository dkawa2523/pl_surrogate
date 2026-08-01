# GEC-CCP NN / Neural-Operator Comparison v2

This is the adopted successor to `gec_ccp_nn_operator_comparison_v1`.
The dataset, 54/11/13 split, preprocessing, loss, validation selection,
and the seven non-POD model recipes are unchanged.  The only model change is
the validation-selected POD coefficient branch.

## Adopted POD-DeepONet recipe

```yaml
model_cfg:
  hidden: [192, 192]
  branch:
    activation: gelu
    activate_last_hidden: true
    descriptor:
      mode: raw
      condition_dim: 4
      dim: 114
      normalization: none
```

The old branch applied GELU only between the two hidden Linear layers.  The
adopted branch applies GELU after both hidden layers.  Validation-only tuning
selected this recipe over train-only descriptor z-score candidates.

Evidence from three seeds is retained under
`runs/gec_ccp_pod_branch_tuned_v2`: validation objective
`0.063080 ± 0.005481`, test quality error `0.001169 ± 0.000242`, and plasma
R2 `0.999010 ± 0.000166`.  Test metrics were read only after the branch was
selected by validation.

## Historical boundary

- `comparison_v1` and its run directory remain the immutable old-branch evidence.
- `comparison_v2` is the canonical recipe for future POD-DeepONet runs.
- Old checkpoints remain loadable and replay their original final-hidden behavior.

## Regenerate

```powershell
.\.venv-torch\Scripts\python.exe `
  scripts/prepare_gec_ccp_nn_operator_comparison_v1_configs.py `
  --out-config-root configs/experimental/gec_ccp_nn_operator_comparison_v2 `
  --run-root runs/gec_ccp_nn_operator_comparison_v2 `
  --protocol-prefix gec_ccp_nn_operator_comparison_v2 `
  --pod-branch-recipe adopted_v2
```

## Run only the adopted POD model

```powershell
.\.venv-torch\Scripts\python.exe scripts/run_gec_ccp_selected_v2_seed_matrix.py `
  --sizes 78 `
  --models deeponet_pod `
  --seeds 411 412 413 `
  --config-root configs/experimental/gec_ccp_nn_operator_comparison_v2 `
  --run-root runs/gec_ccp_nn_operator_comparison_v2 `
  --status-csv runs/gec_ccp_nn_operator_comparison_v2/matrix_status.csv `
  --load-profile normal `
  --stop-on-failure
```

The already completed adopted run is documented in
`runs/gec_ccp_pod_branch_tuned_v2/review.md`; retraining is not needed merely
to use its validated result in the present comparison.
