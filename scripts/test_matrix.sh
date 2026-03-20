#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-torch/bin/python}"
PYTEST_BIN="${PYTEST_BIN:-.venv-torch/bin/pytest}"
export PYTHONPATH="${PYTHONPATH:-src}"

if [[ ! -x "$PYTHON_BIN" || ! -x "$PYTEST_BIN" ]]; then
  echo "Missing virtualenv binaries. Expected $PYTHON_BIN and $PYTEST_BIN" >&2
  exit 1
fi

echo "[lane: torch-disabled] unit"
"$PYTEST_BIN" tests/unit -q -rs

echo "[lane: torch-disabled] integration"
"$PYTEST_BIN" tests/integration -q -rs

echo "[lane: torch-enabled] unit"
PLASMA_SURROGATE_ENABLE_TORCH=1 "$PYTEST_BIN" tests/unit -q -rs

echo "[lane: torch-enabled] integration"
PLASMA_SURROGATE_ENABLE_TORCH=1 "$PYTEST_BIN" tests/integration -q -rs

echo "[matrix] all lanes passed"
