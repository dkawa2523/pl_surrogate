# Head Mode Benchmark Comparison

These smoke configs compare the same grid/operator model with three output head
modes:

- `benchmark_fno_shared_n8.yaml`: shared allvars baseline.
- `benchmark_fno_role_grouped_n8.yaml`: role-grouped heads resolved from
  `target_role_schema.json`.
- `benchmark_fno_custom_groups_n8.yaml`: explicit custom groups under
  `model.output_heads.groups`.

They use the synthetic benchmark dataset with `n_cases: 8` so the configs stay
small and CI-friendly. For n27/n54/n78 style runs, copy one file and change:

- `benchmark.output_dir`
- `benchmark.dataset.n_cases` for synthetic data, or `benchmark.dataset.index_csv`
  for csv/npz datasets such as `index_27.csv`, `index_54.csv`, `index_78.csv`

The benchmark still selects by `surrogate_quality_score`. Group-wise and
region-wise columns are observational and should be compared alongside the
target-wise columns in `leaderboard.csv` / `core_metrics.csv`.
