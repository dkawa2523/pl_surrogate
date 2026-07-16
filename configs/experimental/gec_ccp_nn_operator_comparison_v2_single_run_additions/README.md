# GEC-CCP comparison v2: single-run additions

This directory adds one seed-412 run each for `deeponet_plasma`, `cno`, and
`unetpp`. It is a screening extension of the n78 54/11/13 comparison, not a
hyperparameter search.

The model-specific settings come from the existing ext0520 reference recipes:

- `deeponet_plasma`: 160 epochs, lr 8e-4, latent 48, hidden 96
- `cno`: 80 epochs, lr 8e-4, width 64, 4 layers
- `unetpp`: 120 epochs, lr 4e-4, base channels 32, depth 2

The split, linear target transforms, train-only scalers, 13-channel spatial
input, `plasma_surrogate_v3` loss, and validation-only spatial checkpoint
selection match comparison v2.

Regenerate:

```powershell
.\.venv-torch\Scripts\python.exe `
  scripts/prepare_gec_ccp_nn_operator_comparison_v1_configs.py `
  --out-config-root configs/experimental/gec_ccp_nn_operator_comparison_v2_single_run_additions `
  --run-root runs/gec_ccp_nn_operator_comparison_v2_single_run_additions `
  --protocol-prefix gec_ccp_nn_operator_comparison_v2_single_run_additions `
  --pod-branch-recipe adopted_v2 `
  --seeds 412 `
  --models deeponet_plasma cno unetpp
```
