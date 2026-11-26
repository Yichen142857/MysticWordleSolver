"""Letter cipher and its sampler (README §2)."""

from __future__ import annotations

import string

import numpy as np

ALPHABET = string.ascii_lowercase


class Cipher:
    """A permutation of a–z. `mapping[i]` is the image of ALPHABET[i]."""

    __slots__ = ("mapping", "_enc", "_dec")

    def __init__(self, mapping: str):
        if sorted(mapping) != list(ALPHABET):
            raise ValueError("cipher mapping must be a permutation of a-z")
        self.mapping = mapping
        self._enc = str.maketrans(ALPHABET, mapping)
        self._dec = str.maketrans(mapping, ALPHABET)

    @classmethod
    def identity(cls) -> "Cipher":
        return cls(ALPHABET)

    @classmethod
    def from_swaps(cls, *pairs: str) -> "Cipher":
        """Build a cipher from letter transpositions, e.g. from_swaps("te")."""
        m = list(ALPHABET)
        for a, b in pairs:
            i, j = ALPHABET.index(a), ALPHABET.index(b)
            m[i], m[j] = m[j], m[i]
        return cls("".join(m))

    def encode(self, word: str) -> str:
        return word.translate(self._enc)

    def decode(self, word: str) -> str:
        return word.translate(self._dec)

    def inverse(self) -> "Cipher":
        return Cipher(self.decode(ALPHABET))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cipher) and self.mapping == other.mapping

    def __hash__(self) -> int:
        return hash(self.mapping)

    def __repr__(self) -> str:
        return f"Cipher({self.mapping!r})"


class CipherSampler:
    """Draws ciphers for new games.

    mode="random":   uniform permutation per game (the default game).
    mode="identity": always the identity (ablation: cost of the cipher).
    mode="fixed":    the same cipher every game (ablation: memorizing vs inferring);
                     pass `fixed=Cipher(...)`, or it is drawn once from `fixed_seed`.
    """

    def __init__(self, mode: str = "random", fixed: Cipher | None = None, fixed_seed: int = 0):
        if mode not in ("random", "identity", "fixed"):
            raise ValueError(f"unknown cipher mode {mode!r}")
        self.mode = mode
        if mode == "fixed" and fixed is None:
            fixed = self._draw(np.random.default_rng(fixed_seed))
        self.fixed = fixed

    @staticmethod
    def _draw(rng: np.random.Generator) -> Cipher:
        return Cipher("".join(ALPHABET[i] for i in rng.permutation(26)))

    def sample(self, rng: np.random.Generator) -> Cipher:
        if self.mode == "identity":
            return Cipher.identity()
        if self.mode == "fixed":
            assert self.fixed is not None
            return self.fixed
        return self._draw(rng)
