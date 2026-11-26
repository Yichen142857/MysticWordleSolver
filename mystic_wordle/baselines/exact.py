"""Exact-constraint particle solver (plan §7 step 3 reference solver).

Particles are decoders that are *exactly* consistent with all feedback. Each turn:

1. Old particles are re-checked against the new feedback. Broken ones get a
   short local repair (the word-anchored moves of ParticleSolver). Those that
   still fail are dropped.
2. A CP-SAT model of the whole game (cipher, one word per target, every
   feedback constraint) is solved repeatedly within a time budget. Each solve
   adds a *coverage cut*: for every current particle p, the target words may
   not all lie in p's consistent-word sets. The next solution is therefore a
   word tuple that no particle explains yet, which yields a new decoder.
3. If the model with cuts is infeasible, the particles cover every consistent
   word tuple: the support of the posterior is complete.

Weights: a decoder restricted to the active letters (those that appear in
guesses) has posterior mass proportional to the product over targets of
consistent-word counts (uniform priors on the cipher and on words within a
length). Particles are deduplicated by that restriction and weighted by the
product. Coverage is complete once the model is infeasible. Weights can
still miss other decoders that explain the same tuples, so they are exact
only up to that.

Plain depth-first search over the cipher was tried first and was far too slow:
with a partial cipher, the green/total bounds are loose, and the tree explodes
(10M nodes without a solution at 7 guesses). CP-SAT's clause learning finds
a solution in seconds.
"""

from __future__ import annotations

import time

import numpy as np

from .agents import ParticleSolver
from .kernels import PAD

try:
    from ortools.sat.python import cp_model
    from ortools.util.python.sorted_interval_list import Domain
except ImportError:  # pragma: no cover
    cp_model = None

CP_PARAMS = dict(num_workers=1, cp_model_probing_level=0, linearization_level=0)


class GameModel:
    """CP-SAT model of all feedback so far. Build once per turn, add cuts, solve repeatedly."""

    def __init__(self, search, history):
        if cp_model is None:
            raise ImportError("ExactSolver needs ortools: pip install ortools")
        self.s = search
        t = search.table
        m = self.m = cp_model.CpModel()
        self.errors: list[str] = []  # solver exceptions caught by solve()
        self.c = [m.NewIntVar(0, 25, f"c{x}") for x in range(26)]  # plaintext x -> cipher letter
        m.AddAllDifferent(self.c)
        self.w = []
        self.words = []  # per target: global word indices (local index = position)
        for i, L in enumerate(search.lengths):
            words = np.arange(search.lo[i], search.hi[i])
            self.words.append(words)
            wv = m.NewIntVar(0, len(words) - 1, f"w{i}")
            tv = [m.NewIntVar(0, 25, f"t{i}_{j}") for j in range(L)]
            rows = np.column_stack([np.arange(len(words)), t.pos[words, :L]]).tolist()
            m.AddAllowedAssignments([wv] + tv, rows)
            z = [m.NewIntVar(0, 25, f"z{i}_{j}") for j in range(L)]
            for j in range(L):
                m.AddElement(tv[j], self.c, z[j])
            eq = {}

            def is_(j, a, z=z, eq=eq):
                if (j, a) not in eq:
                    b = m.NewBoolVar("")
                    m.Add(z[j] == a).OnlyEnforceIf(b)
                    m.Add(z[j] != a).OnlyEnforceIf(b.Not())
                    eq[j, a] = b
                return eq[j, a]

            for guess, fb in history:
                f = fb[i]
                if f is None:
                    continue
                gl = [ord(ch) - 97 for ch in guess]
                m.Add(sum(is_(j, gl[j]) for j in range(min(len(gl), L))) == f.green)
                terms = []
                for a in set(gl):
                    cnt = gl.count(a)
                    if cnt == 1:
                        p = m.NewBoolVar("")
                        m.AddMaxEquality(p, [is_(j, a) for j in range(L)])
                        terms.append(p)
                    else:
                        e = m.NewIntVar(0, cnt, "")
                        sv = m.NewIntVar(0, L, "")
                        m.Add(sv == sum(is_(j, a) for j in range(L)))
                        m.AddMinEquality(e, [sv, cnt])
                        terms.append(e)
                m.Add(sum(terms) == f.green + f.yellow)
            self.w.append(wv)
        # Targets are distinct words: same-length targets share a word range.
        by_range = {}
        for i in range(search.N):
            by_range.setdefault((int(search.lo[i]), int(search.hi[i])), []).append(i)
        for (lo, hi), idx in by_range.items():
            if len(idx) > 1:
                m.AddAllDifferent([self.w[i] for i in idx])
        solved_words = {int(search.lo[i]) for i in range(search.N) if search.hi[i] - search.lo[i] == 1}
        for i in range(search.N):
            if self.words[i].shape[0] > 1:
                for local in np.flatnonzero(np.isin(self.words[i], list(solved_words))):
                    m.Add(self.w[i] != int(local))

    def cut(self, p):
        """Forbid word tuples inside particle p's product of consistent-word sets."""
        s, m = self.s, self.m
        lits = []
        for i in range(s.N):
            if self.words[i].shape[0] == 1:
                continue  # solved target: every particle covers it
            local = np.flatnonzero(np.isin(self.words[i], s.candidates(p, i)))
            outside = Domain.from_values(local.tolist()).complement().intersection_with(
                Domain(0, len(self.words[i]) - 1))
            b = m.NewBoolVar("")  # b false -> w_i outside p's set
            m.AddLinearExpressionInDomain(self.w[i], outside).OnlyEnforceIf(b.Not())
            lits.append(b.Not())
        if not lits:
            m.AddBoolOr([False])  # every target solved: nothing left to find
        else:
            m.AddBoolOr(lits)

    def hint(self, d, t):
        self.m.ClearHints()
        dinv = np.argsort(d)
        for x in range(26):
            self.m.AddHint(self.c[x], int(dinv[x]))
        for i in range(self.s.N):
            local = np.flatnonzero(self.words[i] == t[i])
            if local.size:
                self.m.AddHint(self.w[i], int(local[0]))

    def solve(self, seconds, seed):
        """-> (status, decoder, words); status is "found", "infeasible", "timeout" or "error".

        CP-SAT has been seen to raise an internal error on some models
        (IndexError: absl::btree_map::at, ortools 9.15, macOS / Python 3.14),
        always with solution hints set. On any solver exception we retry once
        without hints and with another seed; if that also fails, we return
        "error" and the caller treats it like a timeout.
        """
        t0 = time.time()
        try:
            st, solver = self._run(seconds, seed)
        except Exception as e:  # noqa: BLE001 - CP-SAT raises plain runtime errors
            self.errors.append(f"{type(e).__name__}: {e}")
            self.m.ClearHints()
            try:
                st, solver = self._run(seconds - (time.time() - t0), seed + 1)
            except Exception as e2:  # noqa: BLE001
                self.errors.append(f"retry {type(e2).__name__}: {e2}")
                return "error", None, None
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            cipher = np.array([solver.Value(v) for v in self.c], dtype=np.int64)
            d = np.empty(26, dtype=np.int64)
            d[cipher] = np.arange(26)
            words = np.array([self.words[i][solver.Value(self.w[i])] for i in range(self.s.N)], dtype=np.int64)
            return "found", d, words
        if st == cp_model.INFEASIBLE:
            return "infeasible", None, None
        return "timeout", None, None

    def _run(self, seconds, seed):
        solver = cp_model.CpSolver()
        for k, v in CP_PARAMS.items():
            setattr(solver.parameters, k, v)
        solver.parameters.max_time_in_seconds = max(0.05, seconds)
        solver.parameters.random_seed = seed
        return solver.Solve(self.m), solver


class ExactSolver(ParticleSolver):
    """ParticleSolver whose particles are exactly consistent decoders from CP-SAT.

    cp_seconds:   CP-SAT time budget per turn.
    max_particles: stop adding particles once the pool reaches this size.
    repair_moves: local-repair budget for particles broken by new feedback.
    spawn_copies, spawn_moves: after the CP-SAT phase, each consistent particle
        is copied and moved locally; copies that stay consistent and are new
        become particles. This is cheap, so it fills out each mode CP-SAT finds.
    `self.complete` is True on turns where the pool provably covers every
    consistent word tuple.
    """

    name = "exact"

    def __init__(self, dictionary, known_cipher=None, cp_seconds=10.0, max_particles=64, repair_moves=20,
                 spawn_copies=2, spawn_moves=10, **kw):
        kw.setdefault("n_particles", 16)
        super().__init__(dictionary, known_cipher, **kw)
        self.cp_seconds = cp_seconds
        self.max_particles = max_particles
        self.repair_moves = repair_moves
        self.spawn_copies = spawn_copies
        self.spawn_moves = spawn_moves

    def reset(self, obs, seed):
        super().reset(obs, seed)
        self.complete = False
        self.stats = []  # per turn: dict(particles, solves, complete, seconds)

    # -- particle pool --------------------------------------------------------

    def _keep(self, idx):
        s = self.search
        idx = np.asarray(idx, dtype=np.int64)
        for name in ("D", "T", "v", "nz", "logpi", "logw"):
            setattr(s, name, getattr(s, name)[idx].copy())

    def _add(self, d, t):
        s = self.search
        s.D = np.vstack([s.D, d[None]])
        s.T = np.vstack([s.T, t[None]])
        s.v = np.vstack([s.v, np.zeros((1, s.N), np.int64)])
        s.nz = np.vstack([s.nz, np.zeros((1, s.N), np.int64)])
        s.logpi = np.append(s.logpi, 0.0)
        s.logw = np.append(s.logw, 0.0)
        p = len(s.D) - 1
        s._refresh(p)
        return p

    def _dedup(self):
        s = self.search
        seen, keep = set(), []
        for p in range(len(s.D)):
            key = s.D[p][s.active].tobytes()
            if key not in seen:
                seen.add(key)
                keep.append(p)
        self._keep(keep)

    def _spawn(self):
        s = self.search
        base = np.flatnonzero(s.consistent())
        room = self.max_particles - len(s.D)
        if room <= 0 or len(base) == 0 or self.spawn_copies <= 0:
            return
        parents = np.repeat(base, self.spawn_copies)[:room]
        start = len(s.D)
        self._keep(np.concatenate([np.arange(start), parents]))
        new = np.arange(start, len(s.D))
        s.move(self.spawn_moves, stop_at_zero=False, which=new)
        bad = new[~s.consistent()[new]]
        if len(bad):
            s.move(self.repair_moves, stop_at_zero=True, which=bad)
        self._keep(np.flatnonzero(s.consistent()) if s.consistent().any() else np.arange(len(s.D)))
        self._dedup()

    def _update(self, obs):
        s = self.search
        if s.known:
            s.sync(obs)
            s._refresh_all()
            return
        t0 = time.time()
        if not s.sync(obs):
            return
        s._refresh_all()
        bad = np.flatnonzero(~s.consistent())
        if len(bad):
            s.move(self.repair_moves, stop_at_zero=True, which=bad)
        ok = np.flatnonzero(s.consistent())
        fallback = None
        if len(ok) == 0:
            fallback = int(np.argmin(s.v.sum(1)))
        self._keep(ok if len(ok) else [fallback])
        self._dedup()

        solves, self.complete = 0, False
        model = None
        if s.K > 0:
            model = GameModel(s, obs["history"])
            if len(ok):
                for p in range(len(s.D)):
                    model.cut(p)
            deadline = t0 + self.cp_seconds
            while len(s.D) < self.max_particles or not s.consistent().any():
                left = deadline - time.time()
                if left <= 0:
                    break
                h = int(self.rng.integers(len(s.D)))
                model.hint(s.D[h], s.T[h])
                status, d, words = model.solve(left, int(self.rng.integers(2 ** 31)))
                solves += 1
                if status == "infeasible":
                    self.complete = True
                    break
                if status in ("timeout", "error"):
                    break
                if not s.consistent().any():
                    self._keep([])  # drop the inconsistent fallback
                # Random images for letters no guess has used yet (they do not
                # affect consistency, but they matter for new probe letters).
                free = np.setdiff1d(np.arange(26), s.active)
                if free.size:
                    d[free] = d[free][self.rng.permutation(free.size)]
                p = self._add(d, words)
                assert s.consistent()[p], "CP-SAT solution failed the kernel check"
                model.cut(p)
        self._spawn()
        s.logw = s.logpi.copy()  # weight = sum_i log nz_i for consistent particles
        turn = dict(K=s.K, particles=len(s.D), consistent=int(s.consistent().sum()),
                    solves=solves, complete=self.complete, seconds=round(time.time() - t0, 2))
        if model is not None and model.errors:
            turn["solver_errors"] = model.errors
        self.stats.append(turn)

    def _well_supported(self):
        # With complete coverage the posterior support is known, so trust it.
        return self.complete or super()._well_supported()


def alternative_exists(table, obs, targets, seconds=60.0, seed=0):
    """Identifiability check for a finished (or ongoing) game.

    Is there a word tuple other than the true one that is consistent with all
    feedback in `obs` under some cipher? Solved targets are fixed. Returns
    True (ambiguous), False (the feedback pins the targets down) or None (no
    answer within `seconds`). This measures the information the guesses
    gathered, independent of how well an agent used it.
    """
    from .search import CipherSearch

    s = CipherSearch(table, obs, 1, np.random.default_rng(seed))
    s.sync(obs)
    model = GameModel(s, obs["history"])
    lits = []
    for i, t in enumerate(targets):
        if model.words[i].shape[0] == 1:
            continue
        local = int(np.flatnonzero(model.words[i] == table.index[t])[0])
        b = model.m.NewBoolVar("")
        model.m.Add(model.w[i] != local).OnlyEnforceIf(b)
        lits.append(b)
    if not lits:
        return False
    model.m.AddBoolOr(lits)
    status, _, _ = model.solve(seconds, seed)
    return {"found": True, "infeasible": False, "timeout": None, "error": None}[status]
