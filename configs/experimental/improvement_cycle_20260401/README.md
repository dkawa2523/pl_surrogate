# Improvement Cycle 2026-04-01

This is an archived experimental matrix, not a mainline product route. Keep
generated configs and reports out of the standard docs/tests path unless a test
explicitly exercises archive regeneration.

This directory defines the 12-experiment improvement cycle for:

- `deeponet_plasma` (D1-D6)
- `unetpp_attn / unetpp` (U1-U4)
- `coord_mlp_siren` (C1-C2)

## Materialize Fixtures

```powershell
.\.venv-test\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv-test\Scripts\python.exe experiments\improvement_cycle_20260401\scripts\materialize_improvement_cycle_20260401.py
```

Legacy materializers may write generated fixtures to:

- `tests/fixtures/generated/improvement_cycle_20260401/*.yaml`

Treat those files as reproducible artifacts, not maintained mainline fixtures.

## Legacy Run Example

```powershell
.\.venv-test\Scripts\plasma-surrogate.exe benchmark run --config tests/fixtures/generated/improvement_cycle_20260401/d1_deeponet_epochs240.yaml
```

## Baselines for Gate Evaluation

- Deeponet baseline: `runs/periodic_real_tuned_v83/benchmark_m7_deeponet_isolated_mainline/leaderboard.csv`
- UNet++ Attn baseline: `runs/periodic_real_tuned_v83/benchmark_m7_unetpp_attn_isolated_mainline/leaderboard.csv`
- Coord-MLP-SIREN baseline: `runs/periodic_real_tuned_v83/benchmark_m7_coord_mlp_siren_experimental/leaderboard.csv`

## Evaluate Gates

```powershell
.venv-test\Scripts\python.exe experiments\improvement_cycle_20260401\scripts\evaluate_improvement_cycle_20260401.py `
  --baseline runs/periodic_real_tuned_v83/benchmark_m7_deeponet_isolated_mainline/leaderboard.csv `
  --candidate runs/improvement_cycle_20260401/d6_deeponet_film_fused_residual/leaderboard.csv `
  --track deeponet
```

## Notes

- `D5` and `D6` are materialized from the D3 branch-mode path (`moments`) as the default winner track.
- To switch to the D4 winner track, update `branch_mode`/`sensor_pool_mode` in matrix entries `d5_*` and `d6_*`.
