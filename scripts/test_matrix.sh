#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-torch/bin/python}"
PYTEST_BIN="${PYTEST_BIN:-.venv-torch/bin/pytest}"
PYTEST_COMMON_OPTS=(--import-mode=importlib -q -rs)
PYTEST_SLOW_OPTS=(--import-mode=importlib -q -rs -m benchmark_slow)
PYTEST_TORCH_OPTS=(--import-mode=importlib -q -rs -m torch_runtime)
ALLOWED_SKIP_PATTERN_DISABLED='torch backend disabled|torch runtime unavailable|optuna missing'
ALLOWED_SKIP_PATTERN_ENABLED='optuna missing'

if [[ "$PYTEST_BIN" == *.exe ]]; then
  # WSL -> Windows executable invocation may not propagate env toggles reliably.
  # Keep skip classification explicit for this interop mode.
  ALLOWED_SKIP_PATTERN_ENABLED='torch backend disabled|torch runtime unavailable|optuna missing'
fi

if [[ ! -x "$PYTHON_BIN" || ! -x "$PYTEST_BIN" ]]; then
  echo "Missing virtualenv binaries. Expected $PYTHON_BIN and $PYTEST_BIN" >&2
  exit 1
fi

run_lane() {
  local label="$1"
  local allowed_skip_pattern="$2"
  shift 2
  echo "$label"
  local output
  output="$("$@" 2>&1)" || {
    echo "$output"
    return 1
  }
  echo "$output"
  local unknown_skip
  unknown_skip="$(printf '%s\n' "$output" | grep '^SKIPPED' | grep -Ev "$allowed_skip_pattern" || true)"
  if [[ -n "$unknown_skip" ]]; then
    echo "[matrix] unknown skip reason detected in $label:" >&2
    echo "$unknown_skip" >&2
    return 1
  fi
}

run_lane "[lane: torch-disabled] unit" "$ALLOWED_SKIP_PATTERN_DISABLED" "$PYTEST_BIN" tests/unit "${PYTEST_COMMON_OPTS[@]}"

run_lane "[lane: torch-disabled] integration" "$ALLOWED_SKIP_PATTERN_DISABLED" "$PYTEST_BIN" tests/integration "${PYTEST_COMMON_OPTS[@]}"

export PLASMA_SURROGATE_ENABLE_TORCH=1
if [[ -n "${WSL_INTEROP:-}" ]]; then
  case ":${WSLENV:-}:" in
    *:PLASMA_SURROGATE_ENABLE_TORCH/u:*) ;;
    *) export WSLENV="${WSLENV:+${WSLENV}:}PLASMA_SURROGATE_ENABLE_TORCH/u" ;;
  esac
fi
run_lane "[lane: torch-enabled] unit" "$ALLOWED_SKIP_PATTERN_ENABLED" "$PYTEST_BIN" tests/unit "${PYTEST_COMMON_OPTS[@]}"

run_lane "[lane: torch-enabled] integration" "$ALLOWED_SKIP_PATTERN_ENABLED" "$PYTEST_BIN" tests/integration "${PYTEST_COMMON_OPTS[@]}"

run_lane "[lane: torch-enabled] torch runtime unit" "$ALLOWED_SKIP_PATTERN_ENABLED" "$PYTEST_BIN" tests/unit "${PYTEST_TORCH_OPTS[@]}"

run_lane "[lane: torch-enabled] torch runtime integration" "$ALLOWED_SKIP_PATTERN_ENABLED" "$PYTEST_BIN" tests/integration "${PYTEST_TORCH_OPTS[@]}"

run_lane "[lane: torch-enabled] benchmark slow" "$ALLOWED_SKIP_PATTERN_ENABLED" "$PYTEST_BIN" tests/integration "${PYTEST_SLOW_OPTS[@]}"

run_lane "[lane: optuna] integration smoke" "$ALLOWED_SKIP_PATTERN_ENABLED" "$PYTEST_BIN" tests/integration/test_optimize_optuna_smoke.py "${PYTEST_COMMON_OPTS[@]}"
unset PLASMA_SURROGATE_ENABLE_TORCH

echo "[matrix] all lanes passed"
