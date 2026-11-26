"""Game state and Gym-style environment (README §2, §4, §6)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cipher import Cipher, CipherSampler
from .dictionary import MAX_LEN, MIN_LEN, Dictionary, TargetSampler, is_well_formed
from .scoring import Feedback, score

DEFAULT_N_RANGE = (4, 8)
DEFAULT_MAX_REJECTIONS = 20  # README §9 open item (suggested value)


def default_budget(n: int) -> int:
    """B(N) = N + 16, set by the budget study (README §10): the exact-constraint
    solver wins about 84% of games at this budget, similarly for every N."""
    return n + 16


@dataclass(frozen=True)
class Rewards:
    solve: float = 1.0
    win_bonus: float = 2.0
    valid_guess: float = -0.02
    rejected_guess: float = -0.1


@dataclass(frozen=True)
class HistoryEntry:
    guess: str
    # One entry per target, in target order. None = target was already solved
    # before this guess, so it was not scored.
    feedback: tuple[Feedback | None, ...]


@dataclass
class GameState:
    """Full (hidden + visible) state of one game. Mechanics only: no validation."""

    targets: tuple[str, ...]
    cipher: Cipher
    budget_total: int
    max_rejections: int
    ciphered: tuple[str, ...] = field(init=False)
    budget_left: int = field(init=False)
    solved: list[bool] = field(init=False)
    history: list[HistoryEntry] = field(init=False, default_factory=list)
    rejections: int = 0
    rejection_log: list[tuple[Any, str]] = field(default_factory=list)
    done: bool = False
    won: bool = False

    def __post_init__(self) -> None:
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be distinct")
        self.ciphered = tuple(self.cipher.encode(t) for t in self.targets)
        self.budget_left = self.budget_total
        self.solved = [False] * len(self.targets)

    def apply_guess(self, guess: str) -> tuple[list[Feedback | None], int | None]:
        """Spend one guess. Returns per-target feedback and the newly solved index."""
        assert not self.done and self.budget_left > 0
        self.budget_left -= 1
        feedback: list[Feedback | None] = []
        solved_now = None
        for i, (t, ct) in enumerate(zip(self.targets, self.ciphered)):
            if self.solved[i]:
                feedback.append(None)
                continue
            feedback.append(score(guess, ct))
            if guess == t:  # solve = plaintext match; targets are distinct, so at most one
                self.solved[i] = True
                solved_now = i
        self.history.append(HistoryEntry(guess, tuple(feedback)))
        if all(self.solved):
            self.done, self.won = True, True
        elif self.budget_left == 0:
            self.done = True
        return feedback, solved_now

    def apply_rejection(self, guess: Any, reason: str) -> None:
        self.rejections += 1
        self.rejection_log.append((guess, reason))
        if self.rejections >= self.max_rejections:
            self.done = True


class WordleEnv:
    """Mystic Wordle (cipher Wordle), Gym-style: `reset(seed) -> (obs, info)`,
    `step(guess) -> (obs, reward, terminated, truncated, info)`.

    The observation never contains the cipher or the targets. `env.state` holds
    the full GameState for analysis and tests; agents must not read it.

    Seeding: one seed drives three independent streams (N, targets, cipher), so
    the same seed gives the same N and targets under every cipher mode. This
    keeps identity-vs-random-cipher comparisons paired.
    """

    def __init__(
        self,
        dictionary: Dictionary,
        target_sampler: TargetSampler | None = None,
        cipher_sampler: CipherSampler | None = None,
        n_range: tuple[int, int] = DEFAULT_N_RANGE,
        budget_fn: Callable[[int], int] = default_budget,
        max_rejections: int = DEFAULT_MAX_REJECTIONS,
        rewards: Rewards = Rewards(),
    ):
        self.dictionary = dictionary
        self.target_sampler = target_sampler or TargetSampler(dictionary.targets)
        self.cipher_sampler = cipher_sampler or CipherSampler("random")
        self.n_range = n_range
        self.budget_fn = budget_fn
        self.max_rejections = max_rejections
        self.rewards = rewards
        self.state: GameState | None = None
        self.seed: int | None = None

    # -- Gym API -----------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        *,
        n: int | None = None,
        targets: tuple[str, ...] | list[str] | None = None,
        cipher: Cipher | None = None,
    ) -> tuple[dict, dict]:
        """Start a game. `n`, `targets` and `cipher` override sampling (tests, analyses)."""
        ss = np.random.SeedSequence(seed)
        self.seed = ss.entropy if seed is None else seed
        rng_n, rng_t, rng_c = (np.random.default_rng(s) for s in ss.spawn(3))

        if targets is not None:
            targets = tuple(targets)
            bad = [t for t in targets if not is_well_formed(t)]
            if bad:
                raise ValueError(f"malformed targets {bad}")
            n = len(targets)
        else:
            if n is None:
                lo, hi = self.n_range
                n = int(rng_n.integers(lo, hi + 1))
            targets = self.target_sampler.sample(n, rng_t)
        if cipher is None:
            cipher = self.cipher_sampler.sample(rng_c)

        self.state = GameState(
            targets=targets,
            cipher=cipher,
            budget_total=self.budget_fn(n),
            max_rejections=self.max_rejections,
        )
        return self.observation(), {"seed": self.seed}

    def step(self, guess: Any) -> tuple[dict, float, bool, bool, dict]:
        s = self._require_live()
        reason = self.validate(guess)
        if reason is not None:
            s.apply_rejection(guess, reason)
            info = {"rejected": True, "reason": reason, "solved_now": None, "won": False}
            return self.observation(), self.rewards.rejected_guess, s.done, False, info

        _, solved_now = s.apply_guess(guess)
        reward = self.rewards.valid_guess
        if solved_now is not None:
            reward += self.rewards.solve
        if s.won:
            reward += self.rewards.win_bonus
        info = {"rejected": False, "reason": None, "solved_now": solved_now, "won": s.won}
        return self.observation(), reward, s.done, False, info

    # -- Helpers -----------------------------------------------------------

    def validate(self, guess: Any) -> str | None:
        """Return a rejection reason, or None if the guess is valid."""
        if not isinstance(guess, str):
            return "not a string"
        if not MIN_LEN <= len(guess) <= MAX_LEN:
            return f"length must be {MIN_LEN}-{MAX_LEN}"
        if not is_well_formed(guess):
            return "letters must be lowercase a-z"
        if not self.dictionary.is_valid_guess(guess):
            return "not in dictionary"
        return None

    def observation(self) -> dict:
        """Everything the agent may see (README §2). Immutable values only."""
        s = self._require_state()
        return {
            "n": len(s.targets),
            "target_lengths": tuple(len(t) for t in s.targets),
            "budget_total": s.budget_total,
            "budget_left": s.budget_left,
            "rejections_left": s.max_rejections - s.rejections,
            "solved": tuple(s.solved),
            "history": tuple((h.guess, h.feedback) for h in s.history),
        }

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("call reset() first")
        return self.state

    def _require_live(self) -> GameState:
        s = self._require_state()
        if s.done:
            raise RuntimeError("game is over; call reset()")
        return s
