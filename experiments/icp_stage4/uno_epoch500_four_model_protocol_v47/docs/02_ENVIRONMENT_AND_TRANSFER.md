# Environment and transfer specification

## Reference environment

The current measured environment is:

| Item | Reference |
|---|---|
| Operating system | Windows, PowerShell |
| Python | 3.10.10, 64 bit |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime used by PyTorch | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GiB |
| NumPy | 2.2.6 |
| Training dataset | 957 cases, approximately 6.482 GiB |

Recommended target capacity is one CUDA GPU with at least 16 GiB VRAM,
64 GiB system RAM and 80 GiB free disk. A 32 GiB RAM target may work but is not
the qualified reference. CPU-only training is not a practical 500-epoch path.

## Source transfer rule

Git HEAD alone is not a reproducible source snapshot because the current
working tree contains material uncommitted and untracked ICP implementation
files. Transfer one of the following:

1. Preferred: create a dedicated private commit or archive from the complete
   current working tree and record its archive SHA256.
2. Acceptable: copy the complete source tree while excluding only `.git`,
   virtual environments, `data`, `runs`, and unrelated reports; then transfer
   the required dataset and weights separately.

After transfer, regenerate `SOURCE_INVENTORY.json` and require all listed hashes
to match. Do not treat a successful `git checkout` as sufficient verification.

## Python installation

### Windows PowerShell

```powershell
py -3.10 -m venv .venv-torch
.\.venv-torch\Scripts\python.exe -m pip install --upgrade pip
.\.venv-torch\Scripts\python.exe -m pip install -r requirements-cuda-cu128-py310.txt
.\.venv-torch\Scripts\python.exe -m pip install -e .
$env:PLASMA_SURROGATE_ENABLE_TORCH='1'
$env:PYTHONPATH=(Resolve-Path -LiteralPath 'src').Path
```

### Linux shell

```bash
python3.10 -m venv .venv-torch
.venv-torch/bin/python -m pip install --upgrade pip
.venv-torch/bin/python -m pip install -r requirements-cuda-cu128-py310.txt
.venv-torch/bin/python -m pip install -e .
export PLASMA_SURROGATE_ENABLE_TORCH=1
export PYTHONPATH="$PWD/src"
```

If the target CUDA driver cannot support the pinned cu128 wheel, qualify a new
PyTorch build with the contract smoke before training. Record the exact torch,
CUDA driver and GPU names; do not silently change the requirements file.

## Required data and weights

Transfer at minimum:

```text
data/outputs_icp_stage4_plus_v43_vacuum_q3_v45/
configs/experimental/icp_stage4/conference_continuity_v45/final/conference_sdf/seed_1237.yaml
configs/experimental/icp_stage4/conference_continuity_v45/final/conference_dimension/seed_1237.yaml
configs/experimental/icp_stage4/conference_continuity_v45/initial_weights/conference_sdf_seed1237.npz
configs/experimental/icp_stage4/conference_continuity_v45/initial_weights/conference_dimension_seed1237.npz
experiments/icp_stage4/uno_structure_em_v46/experimental_uno.py
experiments/icp_stage4/uno_structure_em_v46/configs/final/ABC_adaptive_mix/seed_1237.yaml
experiments/icp_stage4/uno_structure_em_v46/configs/initial_weights/ABC_adaptive_mix_seed1237.npz
experiments/icp_stage4/uno_epoch500_four_model_protocol_v47/
```

COMSOL is not needed to train or evaluate against already frozen teacher data.
It is needed only for final validation of newly optimized structures. Record
the COMSOL version, license modules and solver script revision in the
optimization manifest.

## Preflight commands

```powershell
nvidia-smi
.\.venv-torch\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv-torch\Scripts\python.exe -m pytest tests -m "not benchmark_slow"
```

Then run the isolated model-contract smoke and one-epoch smoke for all four
models. The smoke output must use only the dedicated v47 run root.

## Storage behavior

Deterministic preprocessing may be hard-linked into the v47 run directories
when source and target are on the same filesystem and hashes match. If hard
links are unavailable, copy the validated bundles and budget additional disk.
Never point a new run's writable output directly into a v45 or v46 run.

Maintain at least 20 GiB free during training. The recovery mechanism should
retain only the latest resumable optimizer checkpoint and the best selected
model; retaining every optimizer checkpoint can consume tens of GiB.

