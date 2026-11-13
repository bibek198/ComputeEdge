#!/usr/bin/env python3
"""Verifies the retained pre-refactor raw-data manifest against a checkout.

The baseline manifest lists paths relative to the original results directory
(benchmarks/results/). During the refactor that tree moves; this script
locates every listed file by trying each known data root, then checks its
SHA-256 against the retained (never regenerated) value.

Usage: verify_baseline_hashes.py BASELINE.sha256 REPO_ROOT
Exit 0 if every file is found with an identical hash, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Old relative path -> candidate locations, in order. Extended when the data moves.
DATA_ROOTS = [
    "benchmarks/results/{rel}",          # pre-refactor layout
    "data/sessions/orig/{rel}",          # refactored layout: original runs
    "data/sessions/{rel_sessions}",      # refactored layout: confirmation sessions
    "data/derived/{rel}",                # refactored layout: reports / comparisons
    "data/{rel}",                        # refactored layout: top-level files (exclusions.json)
]


def candidates(rel: str) -> list[str]:
    """Possible current locations of a file listed at `rel` in the baseline manifest."""
    rel_sessions = rel[len("sessions/"):] if rel.startswith("sessions/") else rel
    return [r.format(rel=rel, rel_sessions=rel_sessions) for r in DATA_ROOTS]


def sha256(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    manifest, root = Path(sys.argv[1]), Path(sys.argv[2])
    checked, problems = 0, []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line[:64], line[66:]
        found = next((root / c for c in candidates(rel) if (root / c).is_file()), None)
        if found is None:
            problems.append(f"missing: {rel}")
        elif sha256(found) != digest:
            problems.append(f"hash mismatch: {rel} (at {found.relative_to(root)})")
        checked += 1
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{checked} baseline files checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
