"""Belief over the cipher, shared by the consistent guesser and the particle solver.

State: a population of particles, each a decoder d = c^-1 plus one working
word per target, with SMC weights. Weights use the decoder alone: for a
decoder, each target's candidate set is exact (the target-list words of the
right length, or the known word once solved, that are consistent with every
decoded guess), so targets are summed out of the weight. The working words
only drive the search moves (kernels.tuple_moves).
"""

from __future__ import annotations

import numpy as np

from ..cipher import ALPHABET, Cipher
from ..dictionary import Dictionary
from . import kernels as K
from .kernels import MAXLEN, PAD


class WordTable:
    """Encoded word arrays for a word list (sorted by length, then word)."""

    FIELDS = ("pos", "lens", "cnt", "dl", "dc", "nd")

    def __init__(self, words):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        for name, arr in zip(self.FIELDS, self._encode(self.words)):
            setattr(self, name, arr)
        # Contiguous index range per length (the list is sorted by length first).
        self.ranges = {}
        for i, w in enumerate(self.words):
            lo, _ = self.ranges.get(len(w), (i, i))
            self.ranges[len(w)] = (lo, i + 1)

    @staticmethod
    def _encode(words):
        pos, lens, cnt = K.encode_words(words)
        nz = cnt > 0
        order = np.argsort(~nz, axis=1, kind="stable")[:, :MAXLEN]
        dl = order.astype(np.uint8)
        dc = np.take_along_axis(cnt, order, axis=1)
        dl[dc == 0] = PAD
        return pos, lens, cnt, dl, dc, nz.sum(1).astype(np.int64)

    def append(self, word):
        """Add a word that is not in the list (e.g. a solved target outside the prior).
        Appending keeps every existing length range valid."""
        if word in self.index:
            return self.index[word]
        for name, arr in zip(self.FIELDS, self._encode([word])):
            setattr(self, name, np.concatenate([getattr(self, name), arr]))
        self.words.append(word)
        self.index[word] = len(self.words) - 1
        return self.index[word]


def decoder_to_cipher(d) -> Cipher:
    mapping = [""] * 26
    for a in range(26):
        mapping[d[a]] = ALPHABET[a]
    return Cipher("".join(mapping))


class CipherSearch:
    """Particles over decoders, updated by reweighting, resampling and MCMC."""

    def __init__(
        self,
        table: WordTable,
        obs: dict,
        n_particles: int,
        rng: np.random.Generator,
        known_cipher: Cipher | None = None,
        beta: float = 2.0,
        move_params: dict | None = None,
    ):
        self.table = table
        self.rng = rng
        self.beta = beta
        self.known = known_cipher is not None
        self.N = obs["n"]
        self.lengths = obs["target_lengths"]
        self.lo = np.array([table.ranges[L][0] for L in self.lengths], dtype=np.int64)
        self.hi = np.array([table.ranges[L][1] for L in self.lengths], dtype=np.int64)
        self.solved = [False] * self.N

        kmax = obs["budget_total"]
        self.K = 0
        self.g_pos = np.full((kmax, MAXLEN), PAD, dtype=np.uint8)
        self.g_len = np.zeros(kmax, dtype=np.int64)
        self.g_dl = np.full((kmax, MAXLEN), PAD, dtype=np.uint8)
        self.g_dc = np.zeros((kmax, MAXLEN), dtype=np.uint8)
        self.g_nd = np.zeros(kmax, dtype=np.int64)
        self.fb = np.full((kmax, self.N, 2), -1, dtype=np.int64)
        self.active = np.zeros(0, dtype=np.int64)
        self.guessed: set[str] = set()

        if self.known:
            P = 1
            d = np.frombuffer(known_cipher.inverse().mapping.encode(), dtype=np.uint8) - 97
            self.D = d[None, :].astype(np.int64).copy()
        else:
            P = n_particles
            self.D = np.stack([rng.permutation(26) for _ in range(P)]).astype(np.int64)
        self.T = np.stack([[rng.integers(lo, hi) for lo, hi in zip(self.lo, self.hi)] for _ in range(P)]).astype(np.int64)
        self.mp = dict(fit_steps=20000, beta_fit=2.0, patience=4000, beta_word=1.0, beta_accept=1.0, cap=8)
        self.mp.update(move_params or {})
        self.v = np.zeros((P, self.N), dtype=np.int64)
        self.nz = np.zeros((P, self.N), dtype=np.int64)
        self.logpi = np.zeros(P)
        self.logw = np.zeros(P)
        self._refresh_all()

    # -- observation sync ---------------------------------------------------

    def sync(self, obs: dict) -> bool:
        """Absorb new history entries and solves. Returns True if anything changed."""
        hist = obs["history"]
        changed = False
        for guess, feedback in hist[self.K:]:
            k = self.K
            pos, L, dl, dc, nd = K.guess_rows(guess)
            self.g_pos[k], self.g_len[k], self.g_dl[k], self.g_dc[k], self.g_nd[k] = pos, L, dl, dc, nd
            for i, f in enumerate(feedback):
                if f is not None:
                    self.fb[k, i] = f
            self.K += 1
            self.guessed.add(guess)
            changed = True
        for i, s in enumerate(obs["solved"]):
            if s and not self.solved[i]:
                # The solving guess is the last one that still scored target i.
                k = max(k for k in range(self.K) if self.fb[k, i, 0] >= 0)
                word = hist[k][0]
                idx = self.table.append(word)
                self.lo[i], self.hi[i] = idx, idx + 1
                self.T[:, i] = idx
                self.solved[i] = True
                changed = True
        if changed:
            used = np.unique(self.g_pos[: self.K][self.g_pos[: self.K] != PAD])
            self.active = used.astype(np.int64)
        return changed

    def _args(self):
        t = self.table
        return (self.g_pos, self.g_len, self.g_dl, self.g_dc, self.g_nd, self.K, self.fb,
                t.pos, t.lens, t.cnt, self.lo, self.hi)

    def _refresh(self, p):
        K.target_stats(self.D[p], *self._args(), self.v[p], self.nz[p])
        self.logpi[p] = K.log_target(self.v[p], self.nz[p], self.beta)

    def _refresh_all(self):
        for p in range(len(self.D)):
            self._refresh(p)

    # -- SMC step -----------------------------------------------------------

    def update(self, moves: int, stop_at_zero: bool = False) -> None:
        """Reweight particles by the new evidence, resample if needed, then move."""
        old = self.logpi.copy()
        self._refresh_all()
        self.logw += self.logpi - old
        self.logw -= self.logw.max()
        w = np.exp(self.logw)
        w /= w.sum()
        if 1.0 / np.sum(w ** 2) < 0.5 * len(w):
            idx = self._systematic(w)
            for name in ("D", "T", "v", "nz", "logpi"):
                setattr(self, name, getattr(self, name)[idx].copy())
            self.logw[:] = 0.0
        if not self.known:
            self.move(moves, stop_at_zero)

    def move(self, moves: int, stop_at_zero: bool = False, which=None) -> None:
        """Run word-anchored search moves on each particle (or those in `which`)."""
        mp = self.mp
        for p in range(len(self.D)) if which is None else which:
            K.tuple_moves(self.D[p], self.T[p], self.active, moves, mp["fit_steps"], mp["beta_fit"],
                          mp["patience"], mp["beta_word"], mp["beta_accept"], mp["cap"], stop_at_zero, *self._args())
            self._refresh(p)

    def _systematic(self, w):
        P = len(w)
        u = (self.rng.random() + np.arange(P)) / P
        return np.minimum(np.searchsorted(np.cumsum(w), u), P - 1)

    # -- queries ------------------------------------------------------------

    def weights(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def consistent(self) -> np.ndarray:
        """Mask of particles whose decoder admits a consistent word for every target."""
        return (self.v == 0).all(axis=1)

    def candidates(self, p: int, i: int) -> np.ndarray:
        """Word indices for target i under particle p (consistent ones, or the least violating)."""
        return K.best_words(self.D[p], *self._args(), i, self.v[p, i])

    def best_particle(self) -> int:
        score = np.where(self.consistent(), self.logpi, self.logpi - 1e6)
        return int(np.argmax(score))

    def cipher_estimate(self) -> Cipher:
        return decoder_to_cipher(self.D[self.best_particle()])

    def unsolved(self) -> list[int]:
        return [i for i in range(self.N) if not self.solved[i]]
