"""The analysis package's single error type for bad or missing evidence."""

from __future__ import annotations


class DataError(Exception):
    """A data file or session is missing, malformed or not valid for analysis; the message names it."""
