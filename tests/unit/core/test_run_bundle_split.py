from __future__ import annotations

import json

from plasma_surrogate.core.run_bundle import RunBundleLoader


def test_run_bundle_split_random_loads(run_dir):
    split_path = run_dir / "preprocessing" / "split"
    split_path.mkdir(parents=True)
    payload = {"train": ["c0", "c1"], "val": ["c2"], "test": ["c3"]}
    with (split_path / "split_random_v1.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    bundle = RunBundleLoader.load(run_dir)
    split = bundle.split_random()
    assert split == payload

