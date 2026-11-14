"""Pre-registered claims: the registry and its helpers.

Each claim's pass criterion is fixed in code before confirmation data is seen
(docs/analysis_pipeline.md). A claim function takes a Session and returns
(values: dict, passed: bool, note: str).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..io import Session


class Missing(Exception):  # noqa: N818 - control flow ("n/a"), not an error
    """A value a claim needs is absent from the session (the claim is "n/a" there)."""


def need(v):
    """Returns v, or raises Missing if it is None or an empty list."""
    if v is None or (isinstance(v, list) and not v):
        raise Missing()
    return v


@dataclass
class Claim:
    """A pre-registered claim: statement, pass criterion and the function that evaluates it."""

    id: str
    finding: str
    text: str
    criterion: str
    fn: Callable[[Session], tuple[dict, bool, str]]
    headline: dict = field(default_factory=dict)  # value key -> LaTeX macro name
    # Sessions whose values feed the headline macros (e.g. only mixed-load thermal runs).
    headline_if: Callable[[dict], bool] | None = None


CLAIMS: list[Claim] = []


def claim(id, finding, text, criterion, headline=None, headline_if=None):  # noqa: A002 - id is the claim ID
    """Registers the decorated function as a claim in CLAIMS (definition order is report order)."""

    def deco(fn):
        CLAIMS.append(Claim(id, finding, text, criterion, fn, headline or {}, headline_if))
        return fn
    return deco
