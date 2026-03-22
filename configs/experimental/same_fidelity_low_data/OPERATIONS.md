# Same-Fidelity Low-Data Operations

This note defines the operational entrypoints and keeps ablation configs out of
the default production path.

## Canonical benchmark fixtures (tests)

- `tests/fixtures/benchmark_periodic_real_m7_unetpp_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_unetpp_attn_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_ffno_isolated_mainline.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_coord_mlp_fourier_experimental.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_coord_mlp_siren_experimental.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_deeponet_pod_experimental.yaml`
- `tests/fixtures/benchmark_periodic_real_m7_deeponet_isolated_mainline.yaml`

Use these fixture files as the single operational source of truth.
Temporary configs under `tmp_*` (for example `tmp_deeponet_pod/*`) are
analysis-only and must not be referenced in routine benchmark/compare runs.

## Canonical compare entrypoint

- `configs/experimental/same_fidelity_low_data/benchmark_compare_template.yaml`

The template uses current `runs/periodic_real_tuned_v*/.../leaderboard.csv`
paths and stays dynamic-target aware (`objective_metric=auto_primary`).

## Ablations

Files named `benchmark_*_r*.yaml`, `benchmark_*_a*.yaml`, `benchmark_*_b*.yaml`,
`benchmark_*_c*.yaml`, and `benchmark_*_d*.yaml` are ablation configs. They are
kept for analysis only and are not part of the operational baseline set.
