"""Feedback scoring (README §3). Pure functions, no state."""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple


class Feedback(NamedTuple):
    green: int
    yellow: int


def score(guess: str, ciphered_target: str) -> Feedback:
    """Count-based feedback of `guess` against an (already ciphered) target.

    1. Green: positions 1..min(len) where the letters match; both marked used.
    2. Yellow: multiset match of unused guess letters vs unused target letters.

    Works for unequal lengths. Invariant: green + yellow equals the size of the
    multiset intersection of the two strings.
    """
    green = 0
    rest_g: Counter[str] = Counter()
    rest_t: Counter[str] = Counter()
    for g, t in zip(guess, ciphered_target):  # zip stops at min length
        if g == t:
            green += 1
        else:
            rest_g[g] += 1
            rest_t[t] += 1
    m = min(len(guess), len(ciphered_target))
    rest_g.update(guess[m:])
    rest_t.update(ciphered_target[m:])
    yellow = sum((rest_g & rest_t).values())
    return Feedback(green, yellow)
