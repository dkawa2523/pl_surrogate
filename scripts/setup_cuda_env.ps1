param(
    [string]$PythonVersion = "3.10",
    [string]$EnvDir = ".venv-torch",
    [string]$RequirementsFile = "requirements-cuda-cu128-py310.txt"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $RequirementsFile)) {
    throw "Requirements file not found: $RequirementsFile"
}

if (-not (Test-Path $EnvDir)) {
    py -$PythonVersion -m venv $EnvDir
}

$python = Join-Path $EnvDir "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python was not created: $python"
}

& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install -r $RequirementsFile

$env:PYTHONPATH = "src"
$env:PLASMA_SURROGATE_ENABLE_TORCH = "1"

@'
import sys
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
'@ | & $python -

Write-Host ""
Write-Host "CUDA environment is ready."
Write-Host "Use:"
Write-Host "  `$env:PYTHONPATH='src'"
Write-Host "  `$env:PLASMA_SURROGATE_ENABLE_TORCH='1'"
Write-Host "  .\$EnvDir\Scripts\python -m plasma_surrogate.cli.main train --config <config>"
