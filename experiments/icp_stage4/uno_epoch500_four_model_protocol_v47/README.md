# ICP UNO 500-epoch four-model portable protocol v47

日本語の概要と実行判断は [README_JA.md](README_JA.md) を参照してください。

> **Isolation boundary:** this directory is a new, specification-only study.
> It must not edit, regenerate, rename, or overwrite any file under
> `experiments/icp_stage4/conference_continuity_v45`,
> `experiments/icp_stage4/uno_structure_em_v46`,
> `runs/icp_conference_continuity_v45`, or
> `runs/icp_uno_structure_em_v46`.

This package records the complete hand-off specification for sequentially
training and comparing four ICP UNO models for 500 epochs, performing matched
and expressive-structure optimization, evaluating against the frozen COMSOL
dataset, and producing conference figures.

The four models are:

1. `formal_sdf_500`: adopted formal union-SDF model.
2. `formal_dimension_500`: adopted formal seven-scalar Dimension model.
3. `abc_sdf_500`: SDF/EM UNO with multi-resolution processing, separate
   lifting, and adaptive local/global mixing.
4. `abc_dimension_p_500`: the architecture- and magnetic-field-matched
   dimension-parameter control defined in
   [01_MODEL_COMPARISON_CONTRACT.md](docs/01_MODEL_COMPARISON_CONTRACT.md).

The planned execution order is exactly the order above. The expected pure
training time on the current RTX 5060 Ti is about 106 hours. Including feature
generation, standard evaluation, the frozen 75-unknown-structure evaluation,
optimization and figure generation, reserve five to six days.

## Study status

- Protocol freeze: complete.
- Existing v45/v46 evidence inventory: complete.
- 500-epoch implementation and runs: **not started by this document task**.
- This package does not claim any 500-epoch result.
- A launcher must pass all checks in [RUN_CHECKLIST.md](RUN_CHECKLIST.md)
  before long training starts.

## Dedicated paths

Only these new paths may be used by the future implementation:

```text
experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/
runs/icp_uno_epoch500_four_model_v47/
reports/icp_conference_materials/icp_uno_epoch500_four_model_v47/
```

The intended run layout is:

```text
runs/icp_uno_epoch500_four_model_v47/
  formal_sdf_500/seed_1237/
  formal_dimension_500/seed_1237/
  abc_sdf_500/seed_1237/
  abc_dimension_p_500/seed_1237/
  optimization/
    matched_regular/
    expressive_independent_coils/
```

## Reading order

1. [Model and fairness contract](docs/01_MODEL_COMPARISON_CONTRACT.md)
2. [Environment and transfer](docs/02_ENVIRONMENT_AND_TRANSFER.md)
3. [500-epoch training runbook](docs/03_TRAINING_RUNBOOK.md)
4. [Evaluation and optimization](docs/04_EVALUATION_AND_OPTIMIZATION.md)
5. [Conference figures](docs/05_CONFERENCE_FIGURES.md)
6. [Recovery, audit, and acceptance](docs/06_RECOVERY_AUDIT_ACCEPTANCE.md)
7. [Machine-readable protocol](PORTABILITY_MANIFEST.json)
8. [Operator checklist](RUN_CHECKLIST.md)

## Fixed evidence sources

- Dataset: `data/outputs_icp_stage4_plus_v43_vacuum_q3_v45`
- Dataset size: 957 cases, approximately 6.482 GiB.
- Frozen split: 685 train, 136 validation, 136 test.
- Frozen structural test: 75 unknown structures plus 25 matched regular
  anchors.
- Training seed: 1237.
- Optimization seed: 4099.
- Targets: `ne`, `ni`, `Te`, and `phi`.
- No COMSOL response field is permitted as a model input.

Critical input hashes are recorded in [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json).

## Reproducibility warning

The current workspace is based on Git commit
`ad5ca0854c2e754b1193b67ef921ad791f3e4bfa`, but the ICP implementation also
contains uncommitted and untracked source files. Checking out that commit alone
is therefore insufficient. A new environment must receive a frozen copy of the
complete working-tree source plus the files listed in `SOURCE_INVENTORY.json`.
Do not silently replace the working-tree snapshot with repository HEAD.
