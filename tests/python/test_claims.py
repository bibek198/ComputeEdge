"""Tests for computeedge_analysis.claims."""

from __future__ import annotations

import json
from pathlib import Path

from computeedge_analysis import paths
from computeedge_analysis.claims import CLAIMS, run_claims


def test_claims_count():
    assert len(CLAIMS) == 17
    claim_ids = [c.id for c in CLAIMS]
    expected_ids = [f"C{i}" for i in range(1, 18)]
    assert claim_ids == expected_ids


def test_run_claims_matches_baseline(tmp_path: Path):
    out_dir = tmp_path / "claims_report"
    numbers_file = tmp_path / "numbers.tex"

    ret = run_claims(allow_dirty=False, out_dir=out_dir, numbers_path=numbers_file)
    assert ret == 0

    assert (out_dir / "claims.json").exists()
    assert (out_dir / "claims.md").exists()
    assert (out_dir / "claims.csv").exists()
    assert numbers_file.exists()

    # Compare generated claims.json with baseline
    baseline_claims = json.loads((paths.REPO / "docs/verification/baseline/claims.json").read_text())
    generated_claims = json.loads((out_dir / "claims.json").read_text())

    for cid in baseline_claims:
        assert cid in generated_claims
        assert generated_claims[cid]["verdict"] == baseline_claims[cid]["verdict"]
        for sess_name, sdata in baseline_claims[cid]["sessions"].items():
            if sdata is not None:
                assert sess_name in generated_claims[cid]["sessions"]
                gen_sdata = generated_claims[cid]["sessions"][sess_name]
                assert gen_sdata["passed"] == sdata["passed"]
                for val_key, val in sdata["values"].items():
                    if isinstance(val, int | float):
                        assert abs(gen_sdata["values"][val_key] - val) < 1e-6
                    else:
                        assert gen_sdata["values"][val_key] == val
