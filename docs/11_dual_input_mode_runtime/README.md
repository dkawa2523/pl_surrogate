# Dual Input Mode Runtime

This directory documents the runtime contract for selecting how a model receives
case inputs.

The supported modes are:

- `table_only`: models use condition-table inputs only. Geometry is still used as
  output context for evaluation and visualization.
- `table_plus_structure`: models use condition-table inputs plus structure-aware
  features, descriptors, or latent vectors.

The current implementation defaults to strict validation:

- `runtime.input_mode` should be explicit in benchmark and experiment configs.
- `runtime.strict_input_mode` should be `error`.
- `runtime.allow_mode_fallback` should be `false`.
- Checkpoint metadata must match the requested runtime metadata.

Read the files in this order:

1. `00_user_view.md`
2. `10_architecture_and_contract.md`
3. `20_geometry_core_and_feature_lanes.md`
4. `30_preprocess_train_infer_runtime.md`
5. `40_model_policy_and_adapters.md`
6. `50_geom_space_and_optimization.md`
7. `60_yaml_templates_and_examples.md`
8. `70_benchmark_tests_and_rollout.md`
9. `80_migration_notes.md`

New model work should start from `src/plasma_surrogate/core/model_specs.py`,
then add the smallest train, infer, checkpoint, and benchmark hooks needed for
that model family.
