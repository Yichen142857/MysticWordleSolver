"""Word lists, target splits and target sampling (README §5)."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

MIN_LEN, MAX_LEN = 5, 10
LENGTHS = tuple(range(MIN_LEN, MAX_LEN + 1))
_VALID = re.compile(r"^[a-z]+$")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATA = DATA_DIR / "words.tsv"
SPLITS_DIR = DATA_DIR / "splits"      # frozen pools: train.txt, val.txt, test.txt
MANIFEST = DATA_DIR / "manifest.json"  # settings, counts and checksums of the frozen data
# Final data settings (README §5). Every result so far uses them, and changing
# any of them changes the game's difficulty (the budget study would need redoing).
DEFAULT_ZIPF_THRESHOLD = 3.0
DEFAULT_SPLIT = (0.8, 0.1, 0.1)
DEFAULT_SPLIT_SEED = 0
SPLIT_NAMES = ("train", "val", "test")


def is_well_formed(word: str) -> bool:
    return isinstance(word, str) and MIN_LEN <= len(word) <= MAX_LEN and bool(_VALID.match(word))


class Dictionary:
    """Guess dictionary plus the frequency-filtered target list.

    `guesses`: every valid guess word (the full list).
    `targets`: words with zipf >= threshold; the shared prior for all methods.
    `split(...)`: train/val/test target pools, stratified by length.
    """

    def __init__(self, words: Mapping[str, float], zipf_threshold: float = DEFAULT_ZIPF_THRESHOLD):
        bad = [w for w in words if not is_well_formed(str(w))]
        if bad:
            raise ValueError(f"malformed words, e.g. {bad[:5]}")
        self.zipf = {str(w): float(z) for w, z in words.items()}
        self.zipf_threshold = zipf_threshold
        self.guesses: frozenset[str] = frozenset(self.zipf)
        self.targets: tuple[str, ...] = tuple(
            sorted((w for w, z in self.zipf.items() if z >= zipf_threshold), key=lambda w: (len(w), w))
        )

    @classmethod
    def load(cls, path: Path | str = DEFAULT_DATA, zipf_threshold: float = DEFAULT_ZIPF_THRESHOLD) -> "Dictionary":
        with open(path, newline="") as f:
            rows = csv.DictReader(f, delimiter="\t")
            words = {r["word"]: float(r["zipf"]) for r in rows}
        return cls(words, zipf_threshold)

    @classmethod
    def from_words(cls, guesses: Iterable[str], targets: Iterable[str] | None = None) -> "Dictionary":
        """Small hand-built dictionaries for tests. Targets default to all guesses."""
        guesses = {str(w) for w in guesses}
        targets = guesses if targets is None else {str(w) for w in targets}
        if not targets <= guesses:
            raise ValueError("targets must be a subset of guesses")
        return cls({w: (1.0 if w in targets else 0.0) for w in guesses}, zipf_threshold=1.0)

    def is_valid_guess(self, word: str) -> bool:
        return word in self.guesses

    def counts_by_length(self, words: Iterable[str]) -> dict[int, int]:
        out = {n: 0 for n in LENGTHS}
        for w in words:
            out[len(w)] += 1
        return out

    def target_pool(self, name: str, seed: int = DEFAULT_SPLIT_SEED,
                    ratios: Sequence[float] = DEFAULT_SPLIT) -> tuple[str, ...]:
        """The `name` target pool ("train", "val" or "test").

        With the final settings this reads the frozen file data/splits/<name>.txt,
        so the pools cannot drift with numpy or code changes. Any other setting
        (threshold, seed or ratios) is computed with `split()`.
        """
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown pool {name!r}; use one of {SPLIT_NAMES}")
        final = (self.zipf_threshold == DEFAULT_ZIPF_THRESHOLD and seed == DEFAULT_SPLIT_SEED
                 and tuple(ratios) == tuple(DEFAULT_SPLIT))
        path = SPLITS_DIR / f"{name}.txt"
        if final and path.exists():
            words = tuple(path.read_text().split())
            missing = [w for w in words if w not in self.zipf or self.zipf[w] < self.zipf_threshold]
            if missing:
                raise ValueError(f"{path} has words outside the target list, e.g. {missing[:5]}")
            return words
        return self.split(ratios, seed)[name]

    def split(self, ratios: Sequence[float] = DEFAULT_SPLIT, seed: int = DEFAULT_SPLIT_SEED) -> dict[str, tuple[str, ...]]:
        """Deterministic train/val/test split of the target list, stratified by length."""
        if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-9 or min(ratios) < 0:
            raise ValueError("ratios must be three non-negative numbers summing to 1")
        rng = np.random.default_rng(seed)
        pools: dict[str, list[str]] = {k: [] for k in SPLIT_NAMES}
        for n in LENGTHS:
            group = [w for w in self.targets if len(w) == n]
            order = rng.permutation(len(group))
            a = round(ratios[0] * len(group))
            b = a + round(ratios[1] * len(group))
            for k, idx in zip(SPLIT_NAMES, (order[:a], order[a:b], order[b:])):
                pools[k].extend(group[i] for i in idx)
        return {k: tuple(sorted(v, key=lambda w: (len(w), w))) for k, v in pools.items()}


class TargetSampler:
    """Draws N distinct targets from a pool with per-word weights by length group.

    `length_weights[L]` is the total probability mass of length group L in one
    draw (README §5 "target weights"); within a group, words are uniform.
    Default: equal mass per length present in the pool. The final formula is an
    open item; pass a different mapping to change it.
    """

    def __init__(self, pool: Sequence[str], length_weights: Mapping[int, float] | None = None):
        if len(set(pool)) != len(pool):
            raise ValueError("pool contains duplicates")
        self.pool = tuple(pool)
        counts: dict[int, int] = {}
        for w in self.pool:
            counts[len(w)] = counts.get(len(w), 0) + 1
        if length_weights is None:
            length_weights = {n: 1.0 for n in counts}
        p = np.array([length_weights.get(len(w), 0.0) / counts[len(w)] for w in self.pool], dtype=float)
        if p.sum() <= 0:
            raise ValueError("length_weights give zero mass to the pool")
        self.p = p / p.sum()

    def sample(self, n: int, rng: np.random.Generator) -> tuple[str, ...]:
        if n > int(np.count_nonzero(self.p)):
            raise ValueError(f"cannot draw {n} distinct targets from this pool")
        idx = rng.choice(len(self.pool), size=n, replace=False, p=self.p)
        return tuple(self.pool[i] for i in idx)
