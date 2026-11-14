"""Run exclusions: runs kept on disk (raw data is immutable) but left out of analysis.

`<data>/exclusions.json` lists each excluded run with the evidence for excluding it
(data/README.md). A run is never excluded because its result is inconvenient.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ..errors import DataError


def exclusions_file(data_dir: Path | None = None) -> Path:
    """Location of the exclusion list for a data tree."""
    return (data_dir or paths.DATA) / "exclusions.json"


def excluded_runs(data_dir: Path | None = None) -> set[Path]:
    """Resolved paths of the runs listed in exclusions.json (empty if the file does not exist).

    Raises:
        DataError: the file exists but cannot be read or lacks `excluded_runs`.
    """
    data = data_dir or paths.DATA
    path = exclusions_file(data)
    if not path.exists():
        return set()
    try:
        entries = json.loads(path.read_text())["excluded_runs"]
        return {(data / e["run"]).resolve() for e in entries}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise DataError(f"cannot read {path}: {e}") from e
