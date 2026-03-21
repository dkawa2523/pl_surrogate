# Windows CUDA setup

This project does not currently ship a single canonical dependency file, so the
Windows CUDA path is pinned here for reproducibility.

## Validated environment

- OS: Windows
- Python: 3.10
- GPU runtime: NVIDIA CUDA driver detected by `nvidia-smi`
- PyTorch: `2.10.0+cu128`

## One-shot setup

```powershell
Set-Location C:\Users\user\Desktop\pl_surrogate
.\scripts\setup_cuda_env.ps1
```

## Manual setup

```powershell
py -3.10 -m venv .venv-torch
.\.venv-torch\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv-torch\Scripts\python -m pip install -r requirements-cuda-cu128-py310.txt
```

## Validate CUDA

```powershell
@'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
'@ | .\.venv-torch\Scripts\python -
```

## Run project code with torch enabled

```powershell
$env:PYTHONPATH = "src"
$env:PLASMA_SURROGATE_ENABLE_TORCH = "1"
.\.venv-torch\Scripts\python -m plasma_surrogate.cli.main train --config <config>
```

Examples:

```powershell
$env:PYTHONPATH = "src"
$env:PLASMA_SURROGATE_ENABLE_TORCH = "1"
.\.venv-torch\Scripts\pytest tests\unit\core\test_torch_backend.py -q
.\.venv-torch\Scripts\pytest tests\unit\models\test_deeponet_plasma_torch.py -q
```

## Notes

- `python` from the Microsoft Store alias may shadow the real interpreter. Use
  `py -3.10` or the venv interpreter directly.
- The project's docs currently show POSIX-style paths such as
  `.venv-torch/bin/python`; on Windows use `.venv-torch\Scripts\python.exe`.
- Torch-backed code paths require `PLASMA_SURROGATE_ENABLE_TORCH=1`.
