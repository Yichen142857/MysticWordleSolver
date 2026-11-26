"""Algorithmic baselines (README §7 step 5).

All agents share one interface: `reset(obs, seed)` at the start of a game,
then `act(obs) -> guess` each turn. They read only the observation.
`known_cipher` is for the identity-cipher ablation, where the cipher is part
of the rules rather than hidden.
"""

from __future__ import annotations

import numpy as np

from ..cipher import Cipher
from ..dictionary import Dictionary
from . import kernels as K
from .search import CipherSearch, WordTable


class Agent:
    name = "agent"
    # Set by act() in agents that keep a belief: a small JSON-able record of the
    # decision (for imitation learning). See ParticleSolver.act for the fields.
    last_decision: dict | None = None

    def reset(self, obs: dict, seed: int) -> None:
        raise NotImplementedError

    def act(self, obs: dict) -> str:
        raise NotImplementedError

    def cipher_estimate(self) -> Cipher | None:
        """Current best guess at the cipher, or None if the agent keeps no belief."""
        return None


class RandomGuesser(Agent):
    """The floor: uniform dictionary words, never repeating a guess."""

    name = "random"

    def __init__(self, dictionary: Dictionary, pool: str = "guesses"):
        words = dictionary.guesses if pool == "guesses" else dictionary.targets
        self.words = sorted(words)

    def reset(self, obs, seed):
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.words))
        self.t = 0

    def act(self, obs):
        w = self.words[self.order[self.t]]
        self.t += 1
        return w


class _SearchAgent(Agent):
    def __init__(self, dictionary: Dictionary, known_cipher: Cipher | None = None):
        self.dictionary = dictionary
        self.known_cipher = known_cipher
        self.targets = WordTable(dictionary.targets)

    move_params = None

    def _start(self, obs, seed, n_particles, beta):
        self.rng = np.random.default_rng(seed)
        K.seed_rng(int(seed) % (2 ** 32))
        self.search = CipherSearch(self.targets, obs, n_particles, self.rng, self.known_cipher, beta,
                                   self.move_params)

    def cipher_estimate(self):
        return self.search.cipher_estimate()


class ConsistentGuesser(_SearchAgent):
    """Keep one (cipher, words) state; after each guess, search from the previous
    state until it is consistent with all feedback (or the move budget runs out),
    then guess a consistent word for the unsolved target with the fewest
    consistent words under that cipher."""

    name = "consistent"

    def __init__(self, dictionary, known_cipher=None, moves_per_round=100, max_rounds=3, beta=2.0):
        super().__init__(dictionary, known_cipher)
        self.moves_per_round = moves_per_round
        self.max_rounds = max_rounds
        self.beta = beta

    def reset(self, obs, seed):
        self._start(obs, seed, 1, self.beta)

    def act(self, obs):
        s = self.search
        if s.sync(obs):
            s.update(0)
            for _ in range(self.max_rounds):
                if s.known or s.consistent()[0]:
                    break
                s.move(self.moves_per_round, stop_at_zero=True)
        unsolved = s.unsolved()
        order = sorted(unsolved, key=lambda i: (s.v[0, i], s.nz[0, i]))
        for i in order:
            words = [self.targets.words[w] for w in s.candidates(0, i)]
            words = [w for w in words if w not in s.guessed]
            if words:
                guess = words[self.rng.integers(len(words))]
                self.last_decision = {"mode": "consistent", "guess": guess, "target": int(i),
                                      "n_candidates": len(words), "consistent": bool(s.consistent()[0])}
                return guess
        # Every candidate already guessed: fall back to an unused target-list word.
        L = obs["target_lengths"][order[0]]
        lo, hi = self.targets.ranges[L]
        while True:
            w = self.targets.words[self.rng.integers(lo, hi)]
            if w not in s.guessed:
                return w


class ParticleSolver(_SearchAgent):
    """SMC over the cipher with search-move rejuvenation; picks the guess that
    maximizes expected information gain plus a bonus for a likely solve.

    Each turn:
      1. Reweight particles by the new feedback, resample when the effective
         sample size drops below half, then run `moves` word-anchored search
         moves per particle (kernels.tuple_moves). Particles that are still
         inconsistent get up to `rescue_rounds` x `rescue_moves` more moves,
         stopping as soon as they become consistent.
      2. Draw `n_hypotheses` joint hypotheses (particle, one consistent word per
         unsolved target) and the ciphered target strings they imply.
      3. Score `n_probes` random dictionary words plus every hypothesised word by
         H(feedback) + kappa * P(solve), and guess the best.
    Overrides: guess a word immediately when P(solve) >= `commit` and at least
    `min_support` particles are consistent (a few particles in one mode are
    overconfident), and only go for solves once the budget left equals the
    number of unsolved targets.
    """

    name = "particle"

    def __init__(
        self,
        dictionary,
        known_cipher=None,
        n_particles=16,
        moves=5,
        rescue_moves=40,
        n_hypotheses=256,
        n_probes=1000,
        kappa=3.0,
        commit=0.95,
        min_support=4,
        beta=2.0,
        rescue_rounds=2,
        move_params=None,
    ):
        super().__init__(dictionary, known_cipher)
        self.guess_table = WordTable(sorted(dictionary.guesses, key=lambda w: (len(w), w)))
        self.n_particles = n_particles
        self.moves = moves
        self.rescue_moves = rescue_moves
        self.move_params = move_params
        self.n_hypotheses = n_hypotheses
        self.n_probes = n_probes
        self.kappa = kappa
        self.commit = commit
        self.min_support = min_support
        self.beta = beta
        self.rescue_rounds = rescue_rounds

    def reset(self, obs, seed):
        self._start(obs, seed, self.n_particles, self.beta)

    # -- belief update --------------------------------------------------------

    def _update(self, obs):
        s = self.search
        if not s.sync(obs):
            return
        s.update(self.moves)
        for _ in range(self.rescue_rounds):
            bad = np.flatnonzero(~s.consistent())
            if s.known or len(bad) == 0:
                break
            s.move(self.rescue_moves, stop_at_zero=True, which=bad)

    # -- hypotheses -----------------------------------------------------------

    def _hypotheses(self, unsolved):
        """-> word indices (S x U) into the target table, ciphered arrays, weights."""
        s = self.search
        ok = s.consistent()
        pool = np.flatnonzero(ok) if ok.any() else np.arange(len(s.D))
        w = s.weights()[pool]
        w = w / w.sum()
        per = max(1, self.n_hypotheses // len(pool))
        S, U = len(pool) * per, len(unsolved)
        words = np.empty((S, U), dtype=np.int64)
        h_pos = np.full((S, U, K.MAXLEN), K.PAD, dtype=np.uint8)
        h_len = np.zeros((S, U), dtype=np.int64)
        h_cnt = np.zeros((S, U, 26), dtype=np.uint8)
        h_w = np.repeat(w / per, per)
        cache = {}
        t = self.targets
        for a, p in enumerate(pool):
            d = s.D[p]
            dinv = np.argsort(d)  # plaintext letter -> cipher letter
            key = d.tobytes()
            for u, i in enumerate(unsolved):
                if (key, i) not in cache:
                    cache[key, i] = s.candidates(p, i)
                cands = cache[key, i]
                picks = cands[self.rng.integers(len(cands), size=per)]
                for r, wi in enumerate(picks):
                    row = a * per + r
                    L = t.lens[wi]
                    words[row, u] = wi
                    h_pos[row, u, :L] = dinv[t.pos[wi, :L]]
                    h_len[row, u] = L
                    h_cnt[row, u] = t.cnt[wi][d]
        return words, h_pos, h_len, h_cnt, h_w

    def _well_supported(self):
        """Enough consistent particles that a high P(solve) is not one mode's overconfidence."""
        s = self.search
        support = len(s.D) if s.known else int(s.consistent().sum())
        return support >= min(self.min_support, len(s.D))

    # -- action ---------------------------------------------------------------

    def act(self, obs):
        self._update(obs)
        s = self.search
        unsolved = s.unsolved()
        words, h_pos, h_len, h_cnt, h_w = self._hypotheses(unsolved)

        p_solve: dict[str, float] = {}
        for row in range(words.shape[0]):
            for wi in set(words[row].tolist()):
                wd = self.targets.words[wi]
                p_solve[wd] = p_solve.get(wd, 0.0) + h_w[row]
        for g in s.guessed:
            p_solve.pop(g, None)

        best_word, best_p = max(p_solve.items(), key=lambda kv: kv[1], default=(None, 0.0))
        confident = best_p >= self.commit and self._well_supported()
        decision = {
            "support": int(len(s.D) if s.known else s.consistent().sum()),
            "complete": bool(getattr(self, "complete", False)),
            "marginals": self._marginals(words, h_w, unsolved),
        }
        if best_word is not None and (confident or obs["budget_left"] <= len(unsolved)):
            decision.update(mode="commit" if confident else "forced", guess=best_word, p_solve=round(best_p, 4),
                            top=[[w, round(p, 4)] for w, p in sorted(p_solve.items(), key=lambda kv: -kv[1])[:5]])
            self.last_decision = decision
            return best_word

        gt = self.guess_table
        probes = self.rng.choice(len(gt.words), size=min(self.n_probes, len(gt.words)), replace=False)
        cand = {int(i) for i in probes}
        cand.update(gt.index[w] for w in p_solve if w in gt.index)
        cand = np.array(sorted(i for i in cand if gt.words[i] not in s.guessed), dtype=np.int64)
        H = K.feedback_entropy(gt.pos[cand], gt.lens[cand], gt.dl[cand], gt.dc[cand], gt.nd[cand],
                               h_pos, h_len, h_cnt, h_w)
        bonus = np.array([p_solve.get(gt.words[i], 0.0) for i in cand])
        util = H + self.kappa * bonus
        order = np.argsort(-util, kind="stable")  # stable: ties break like argmax
        best = int(order[0])
        decision.update(mode="probe", guess=gt.words[cand[best]], entropy=round(float(H[best]), 4),
                        p_solve=round(float(bonus[best]), 4),
                        top=[[gt.words[cand[i]], round(float(H[i]), 4), round(float(bonus[i]), 4)] for i in order[:5]])
        self.last_decision = decision
        return gt.words[cand[best]]

    def _marginals(self, words, h_w, unsolved, k=5):
        """Top-k posterior words per unsolved target (target index -> [[word, prob], ...])."""
        out = {}
        for u, i in enumerate(unsolved):
            mass: dict[int, float] = {}
            for row in range(words.shape[0]):
                mass[words[row, u]] = mass.get(words[row, u], 0.0) + h_w[row]
            top = sorted(mass.items(), key=lambda kv: -kv[1])[:k]
            out[str(i)] = [[self.targets.words[w], round(float(p), 4)] for w, p in top]
        return out
