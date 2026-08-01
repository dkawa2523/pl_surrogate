from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mainline_docs_scripts_and_templates_do_not_expose_removed_research_knobs() -> None:
    banned = (
        "log_ne",
        "log_ni",
        "pow10",
        "exp10",
        "region_balance",
        "spatial_consistency",
        "target_region_by_var",
        "positive_penalty",
        "relative_weighting",
        "external_operator",
        "supervised.base",
        "supervised.delta",
        "delta_by_var",
        "case_spatial_feature_pack",
        "docs/reports",
        "reports/",
    )
    roots = [
        ROOT / "docs",
        ROOT / "configs" / "benchmarkrun_ext0520",
        ROOT / "scripts",
    ]
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for token in banned:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT).as_posix()}:{token}")
    assert hits == []
