# 500-epoch sequential training runbook

## Required execution sequence

Run exactly one training process at a time on the GPU:

1. `formal_sdf_500`
2. `formal_dimension_500`
3. `abc_sdf_500`
4. `abc_dimension_p_500`

Do not run two models concurrently. The measured elapsed-time baselines are:

| Model | Existing measurement | Linear 500-epoch estimate |
|---|---:|---:|
| Formal Dimension | 50 epochs in 63.07 min | 10.51 h |
| Formal SDF | 50 epochs in 75.64 min | 12.61 h |
| ABC-SDF | 15 epochs in 74.79 min | 41.55 h |
| ABC-Dimension-P | architecture-matched estimate | 40-43 h |

Pure sequential training is approximately 105-108 hours. Reserve five to six
days for the full workflow.

## Configuration generation

The future isolated launcher must clone the adopted configurations in memory
and write generated YAML only below:

```text
experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/configs/generated/
```

It must apply these common overrides:

```yaml
seed: 1237
train:
  u_no:
    epochs: 500
    batch_size_cases: 16
  unet_like:
    shuffle_cases: true
split:
  fixed_interp:
    path: data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/split_membership.json
    allow_unassigned: false
```

Early stopping must be disabled. Selection of the best validation checkpoint
is still allowed, but every model must complete all 500 optimization epochs.

## Required launcher interface

The isolated implementation should expose these commands in the new study
directory. These are the canonical commands for the receiving environment;
they are not evidence that the launcher already exists.

```powershell
$python='.\.venv-torch\Scripts\python.exe'
$study='experiments\icp_stage4\uno_epoch500_four_model_protocol_v47'

& $python "$study\verify_portability.py"
& $python "$study\smoke_model_contract.py"
& $python "$study\run_study.py" --stage generate --epochs 500
& $python "$study\run_study.py" --stage smoke
& $python "$study\run_study.py" --stage train --epochs 500 --sequential
& $python "$study\run_study.py" --stage evaluate
& $python "$study\run_study.py" --stage optimize
& $python "$study\run_study.py" --stage figures
& $python "$study\run_study.py" --stage audit
```

Before implementation, each missing command is a blocker. Do not substitute a
v45/v46 launcher that writes into an adopted run root.

## Epoch and optimizer contract

- AdamW.
- Learning rate `0.0003`.
- Weight decay `0.0001`.
- Betas `(0.9, 0.999)`.
- Epsilon `1e-8`.
- Cosine schedule over the full 500 epochs.
- Warmup unchanged from the adopted configuration.
- Batch size 16.
- Fixed random seed 1237.
- No gradient accumulation unless all four models use it and a separate
  equivalence smoke proves the same effective updates.
- No mixed precision unless qualified for all four models as a separate,
  recorded runtime-only change.

Do not implement 500 epochs as ten independent 50-epoch jobs. Restarting AdamW
and the cosine schedule changes the experiment.

## Validation and selection

The formal models retain their adopted best-validation spatial selection
semantics. The ABC pair must use the same selection cadence and warmup as each
other. Record:

- every epoch's training and validation loss;
- learning rate;
- selected epoch and selection score;
- elapsed seconds;
- gradient diagnostics;
- recovery checkpoint epoch;
- final and selected-checkpoint hashes.

Validation selection does not authorize early termination.

## Resume requirement

Before starting a multi-day run, the isolated launcher must save a resumable
checkpoint every 25 epochs containing:

- model state;
- optimizer state;
- current epoch;
- complete cosine-schedule position;
- NumPy RNG state;
- PyTorch CPU RNG state;
- all CUDA RNG states;
- training history and best-validation state;
- resolved configuration hash and dataset manifest hash.

Retain `resume_latest` plus the selected best checkpoint. Resume must reject a
different config, dataset or model implementation hash. This recovery feature
is runtime infrastructure and must not modify the formal model implementation.

## Completion sentinel

A model is complete only if all of the following exist:

```text
leaderboard.csv
resolved_config.yaml
models/u_no/eval_protocol/interp/checkpoints/weights.npz
models/u_no/eval_protocol/interp/checkpoints/meta.json
models/u_no/eval_protocol/interp/train/scalars/metrics.csv
models/u_no/eval_protocol/interp/train/scalars/progress_latest.json
RUN_COMPLETE.json
```

`RUN_COMPLETE.json` must state `epochs_completed: 500`, the selected epoch,
final hashes, elapsed time and `status: complete`. File existence without the
500-epoch assertion is insufficient.

