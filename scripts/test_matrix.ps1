param(
    [string]$PythonBin = ".venv-torch\\Scripts\\python.exe",
    [string]$PytestBin = ".venv-torch\\Scripts\\pytest.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonBin) -or -not (Test-Path $PytestBin)) {
    Write-Error "Missing virtualenv binaries. Expected $PythonBin and $PytestBin"
}

$pytestCommonOpts = @("--import-mode=importlib", "-q", "-rs")
$pytestSlowOpts = @("--import-mode=importlib", "-q", "-rs", "-m", "benchmark_slow")
$pytestTorchOpts = @("--import-mode=importlib", "-q", "-rs", "-m", "torch_runtime")
$allowedSkipPatternDisabled = "torch backend disabled|torch runtime unavailable|optuna missing"
$allowedSkipPatternEnabled = "optuna missing"

function Invoke-PytestLane {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Runner,
        [Parameter(Mandatory = $true)][string]$AllowedSkipPattern
    )
    Write-Host $Label
    $output = & $Runner 2>&1
    $output | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "Lane failed: $Label"
    }
    $unknownSkips = @($output | Where-Object { $_ -match "^SKIPPED" -and $_ -notmatch $AllowedSkipPattern })
    if ($unknownSkips.Count -gt 0) {
        Write-Error "[matrix] unknown skip reason detected in $Label"
        $unknownSkips | ForEach-Object { Write-Error $_ }
    }
}

Invoke-PytestLane -Label "[lane: torch-disabled] unit" -Runner { & $PytestBin tests/unit @pytestCommonOpts } -AllowedSkipPattern $allowedSkipPatternDisabled
Invoke-PytestLane -Label "[lane: torch-disabled] integration" -Runner { & $PytestBin tests/integration @pytestCommonOpts } -AllowedSkipPattern $allowedSkipPatternDisabled

$originalTorchFlag = $env:PLASMA_SURROGATE_ENABLE_TORCH
try {
    $env:PLASMA_SURROGATE_ENABLE_TORCH = "1"
    Invoke-PytestLane -Label "[lane: torch-enabled] unit" -Runner { & $PytestBin tests/unit @pytestCommonOpts } -AllowedSkipPattern $allowedSkipPatternEnabled
    Invoke-PytestLane -Label "[lane: torch-enabled] integration" -Runner { & $PytestBin tests/integration @pytestCommonOpts } -AllowedSkipPattern $allowedSkipPatternEnabled
    Invoke-PytestLane -Label "[lane: torch-enabled] torch runtime unit" -Runner { & $PytestBin tests/unit @pytestTorchOpts } -AllowedSkipPattern $allowedSkipPatternEnabled
    Invoke-PytestLane -Label "[lane: torch-enabled] torch runtime integration" -Runner { & $PytestBin tests/integration @pytestTorchOpts } -AllowedSkipPattern $allowedSkipPatternEnabled
    Invoke-PytestLane -Label "[lane: torch-enabled] benchmark slow" -Runner { & $PytestBin tests/integration @pytestSlowOpts } -AllowedSkipPattern $allowedSkipPatternEnabled
}
finally {
    if ($null -eq $originalTorchFlag) {
        Remove-Item Env:PLASMA_SURROGATE_ENABLE_TORCH -ErrorAction SilentlyContinue
    }
    else {
        $env:PLASMA_SURROGATE_ENABLE_TORCH = $originalTorchFlag
    }
}

Invoke-PytestLane -Label "[lane: optuna] integration smoke" -Runner {
    & $PytestBin tests/integration/test_optimize_optuna_smoke.py @pytestCommonOpts
} -AllowedSkipPattern $allowedSkipPatternEnabled

Write-Host "[matrix] all lanes passed"
