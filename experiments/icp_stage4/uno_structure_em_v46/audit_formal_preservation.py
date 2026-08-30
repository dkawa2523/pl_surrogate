#!/usr/bin/env python
"""Verify that isolated v46 execution did not mutate formal checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = HERE / "formal_reference_snapshot.json"
REPORT = ROOT / "reports/icp_conference_materials/uno_structure_em_v46/formal_preservation_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    checks = []
    for name, payload in expected.items():
        if not isinstance(payload, dict) or "path" not in payload:
            continue
        path = ROOT / payload["path"]
        actual_hash = _sha256(path)
        actual_bytes = path.stat().st_size
        checks.append(
            {
                "reference": name,
                "path": payload["path"],
                "expected_sha256": payload["sha256"],
                "actual_sha256": actual_hash,
                "expected_bytes": int(payload["bytes"]),
                "actual_bytes": int(actual_bytes),
                "unchanged": actual_hash == payload["sha256"] and actual_bytes == int(payload["bytes"]),
            }
        )
    result = {
        "formal_models_unchanged": bool(checks) and all(check["unchanged"] for check in checks),
        "checks": checks,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["formal_models_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
