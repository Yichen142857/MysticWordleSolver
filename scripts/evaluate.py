"""Evaluate baselines on held-out games (README §8).

Every (n, game index) pair maps to one seed, so all agents and both cipher
modes play the same targets: comparisons are paired.

Examples:
  python scripts/evaluate.py --agents random consistent particle --n 4 5 6 7 8 --games 50
  python scripts/evaluate.py --agents particle --cipher identity --games 100 --workers 2

Writes one JSON line per game to --out (default results/<timestamp>.jsonl) and
prints a summary table. `summarize` can be re-run on saved files:
  python scripts/evaluate.py --summarize results/a.jsonl results/b.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mystic_wordle import Cipher, CipherSampler, Dictionary, TargetSampler, WordleEnv  # noqa: E402
from mystic_wordle.baselines import ConsistentGuesser, ParticleSolver, RandomGuesser  # noqa: E402
from mystic_wordle.baselines.exact import ExactSolver, alternative_exists  # noqa: E402
from mystic_wordle.baselines.search import WordTable  # noqa: E402

AGENTS = {"random": RandomGuesser, "consistent": ConsistentGuesser, "particle": ParticleSolver,
          "exact": ExactSolver}
SEED_STRIDE = 1_000_003  # seed = seed_base + n * SEED_STRIDE + game index


def game_seed(seed_base: int, n: int, g: int) -> int:
    return seed_base + n * SEED_STRIDE + g


# -- worker -------------------------------------------------------------------

_W: dict = {}


def _init(cfg):
    d = Dictionary.load(zipf_threshold=cfg["zipf"])
    pool = d.target_pool(cfg["split"], seed=cfg["split_seed"])
    mode = cfg["cipher"]
    extra = cfg.get("budget_extra", 16)
    _W["env"] = WordleEnv(d, TargetSampler(pool), CipherSampler(mode), budget_fn=lambda n: n + extra)
    known = Cipher.identity() if mode == "identity" else None
    agents = {}
    for name in cfg["agents"]:
        kwargs = {} if name == "random" else {"known_cipher": known}
        kwargs.update(cfg["agent_kwargs"].get(name, {}))
        agents[name] = AGENTS[name](d, **kwargs)
    _W["agents"] = agents
    _W["cfg"] = cfg
    if cfg.get("identifiability"):
        _W["table"] = WordTable(d.targets)


def cipher_accuracy(est: Cipher | None, true: Cipher, targets) -> float | None:
    """Fraction of target letters whose cipher image the agent has right."""
    if est is None:
        return None
    letters = sorted(set("".join(targets)))
    return sum(est.encode(x) == true.encode(x) for x in letters) / len(letters)


def replay(env, seed, n, guesses):
    """Re-play a guess sequence; return the observation after it."""
    obs, _ = env.reset(seed, n=n)
    for g in guesses:
        obs, *_ = env.step(g)
    return obs


def identifiability(table, env, seed, n, guesses, seconds):
    """Is the target tuple pinned down by the feedback?

    probe_budget: after the first B - N guesses. If the targets are unique
        here, an agent with perfect inference could solve them all with the N
        guesses left, so the game was winnable with this probe sequence.
    end: after all guesses (lost games only).
    Values: False = unique (identified), True = ambiguous, None = no answer in time.
    """
    obs, _ = env.reset(seed, n=n)
    targets = env.state.targets
    out = {}
    k = obs["budget_total"] - n
    out["alt_exists_probe_budget"] = alternative_exists(table, replay(env, seed, n, guesses[:k]), targets, seconds)
    obs = replay(env, seed, n, guesses)
    if not env.state.won:
        out["alt_exists"] = alternative_exists(table, obs, targets, seconds)
    return out


def play(job):
    """Play one game. A crash is recorded as an error row instead of killing the run."""
    try:
        return _play(job)
    except Exception:
        import traceback
        name, n, g = job
        tb = traceback.format_exc()
        print(f"ERROR in game agent={name} n={n} game={g}:\n{tb}", file=sys.stderr, flush=True)
        return {"agent": name, "cipher": _W["cfg"]["cipher"], "n": n, "game": g,
                "seed": game_seed(_W["cfg"]["seed_base"], n, g), "error": tb}


def _play(job):
    name, n, g = job
    cfg, env, agent = _W["cfg"], _W["env"], _W["agents"][name]
    seed = game_seed(cfg["seed_base"], n, g)
    obs, _ = env.reset(seed, n=n)
    agent.reset(obs, seed)
    t0 = time.time()
    acc_trace = []
    decisions = [] if cfg.get("record_decisions") else None
    done = False
    while not done:
        agent.last_decision = None
        guess = agent.act(obs)
        if decisions is not None:
            decisions.append(agent.last_decision)
        acc_trace.append(cipher_accuracy(agent.cipher_estimate(), env.state.cipher, env.state.targets))
        obs, _, done, _, _ = env.step(guess)
    s = env.state
    extra = {"guess_list": [h.guess for h in s.history]}
    if decisions is not None:
        extra["decisions"] = decisions
    if cfg.get("identifiability"):
        extra.update(identifiability(_W["table"], env, seed, n, extra["guess_list"], cfg["identifiability"]))
    if hasattr(agent, "stats"):
        extra["turn_stats"] = agent.stats
    return {**extra,
        "agent": name, "cipher": cfg["cipher"], "n": n, "game": g, "seed": seed,
        "won": s.won, "solved": list(s.solved), "n_solved": sum(s.solved),
        "guesses": len(s.history), "budget": s.budget_total, "rejections": s.rejections,
        "target_lengths": [len(t) for t in s.targets], "targets": list(s.targets),
        "cipher_acc": acc_trace, "seconds": round(time.time() - t0, 2),
    }


# -- summary ------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def mean_ci(xs, z: float = 1.96):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"),) * 3
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = z * sd / math.sqrt(len(xs))
    return m, m - h, m + h


def summarize(rows) -> str:
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    groups = defaultdict(list)
    for r in rows:
        groups[r["agent"], r["cipher"], r["n"]].append(r)
    out = ["| agent | cipher | N | games | full solve (95% CI) | per-target solve (95% CI) | guesses used | rejections/game | final cipher acc | identified after B-N | identified at end | s/game |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (a, c, n), rs in sorted(groups.items()):
        fs = wilson(sum(r["won"] for r in rs), len(rs))
        # Per-target rate: CI over games (targets within a game are correlated).
        pt = mean_ci([r["n_solved"] / r["n"] for r in rs])
        pt = (pt[0], max(0.0, pt[1]), min(1.0, pt[2]))
        gu = mean_ci([r["guesses"] for r in rs])
        rej = mean_ci([r["rejections"] for r in rs])
        accs = [r["cipher_acc"][-1] for r in rs if r["cipher_acc"] and r["cipher_acc"][-1] is not None]
        def ident(key, won_counts):
            if not any(key in r for r in rs):
                return "-"  # no identifiability check ran for this group
            rows_ = [r for r in rs if key in r or (won_counts and r["won"])]
            known = [r for r in rows_ if (won_counts and r["won"]) or r.get(key) is not None]
            uniq = sum((won_counts and r["won"]) or r.get(key) is False for r in known)
            return f"{uniq}/{len(known)}" + (f" (+{len(rows_) - len(known)} unknown)" if len(known) < len(rows_) else "")
        id_probe = ident("alt_exists_probe_budget", False)
        id_txt = ident("alt_exists", True)
        acc = f"{sum(accs) / len(accs):.2f}" if accs else "-"
        sec = sum(r["seconds"] for r in rs) / len(rs)
        out.append(f"| {a} | {c} | {n} | {len(rs)} | {fs[0]:.2f} ({fs[1]:.2f}-{fs[2]:.2f}) | "
                   f"{pt[0]:.2f} ({pt[1]:.2f}-{pt[2]:.2f}) | {gu[0]:.1f} | {rej[0]:.1f} | {acc} | {id_probe} | {id_txt} | {sec:.0f} |")

    by_len = defaultdict(lambda: [0, 0])
    for r in rows:
        for L, ok in zip(r["target_lengths"], r["solved"]):
            cell = by_len[r["agent"], r["cipher"], L]
            cell[0] += ok
            cell[1] += 1
    lengths = sorted({k[2] for k in by_len})
    out += ["", "Per-target solve rate by target length:", "",
            "| agent | cipher | " + " | ".join(f"L={L}" for L in lengths) + " |",
            "|---|---|" + "---|" * len(lengths)]
    for a, c in sorted({k[:2] for k in by_len}):
        cells = []
        for L in lengths:
            k, m = by_len.get((a, c, L), (0, 0))
            cells.append(f"{k / m:.2f} ({m})" if m else "-")
        out.append(f"| {a} | {c} | " + " | ".join(cells) + " |")
    # Win rate at smaller budgets, read off the guess counts of won games. The
    # agents only change behaviour when the budget left nears the number of
    # unsolved targets, so this approximates (slightly underestimates) a rerun
    # at that budget.
    extras = sorted({e for (a, c, n), rs in groups.items() for r in rs
                     for e in range(10, r["budget"] - n + 1, 2)})
    if extras and any(max(r["budget"] for r in rs) - n > 10 for (a, c, n), rs in groups.items()):
        out += ["", "Wins within a smaller budget B = N + e (from guess counts of won games):", "",
                "| agent | cipher | N | games | " + " | ".join(f"e={e}" for e in extras) + " |",
                "|---|---|---|---|" + "---|" * len(extras)]
        for (a, c, n), rs in sorted(groups.items()):
            cells = []
            for e in extras:
                if e > min(r["budget"] for r in rs) - n:
                    cells.append("-")
                    continue
                k = sum(r["won"] and r["guesses"] <= n + e for r in rs)
                p, lo, hi = wilson(k, len(rs))
                cells.append(f"{p:.2f} ({lo:.2f}-{hi:.2f})")
            out.append(f"| {a} | {c} | {n} | {len(rs)} | " + " | ".join(cells) + " |")
    if errors:
        out += ["", f"{len(errors)} game(s) crashed and are excluded (rows with an 'error' field): "
                + ", ".join(f"{r['agent']} n={r['n']} game={r['game']}" for r in errors)]
    return "\n".join(out)


# -- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", choices=list(AGENTS), default=list(AGENTS))
    ap.add_argument("--cipher", choices=["random", "identity"], default="random")
    ap.add_argument("--n", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    ap.add_argument("--games", type=int, default=20, help="games per N")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--zipf", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--agent-kwargs", default="{}", help='JSON, e.g. \'{"particle": {"n_particles": 32}}\'')
    ap.add_argument("--budget-extra", type=int, default=16, help="budget B = N + this (default 16, the game's rule)")
    ap.add_argument("--record-decisions", action="store_true",
                    help="store each turn's decision record (mode, guess, top alternatives, per-target "
                         "posterior top words) for imitation learning; rows get much larger")
    ap.add_argument("--identifiability", type=float, default=0.0, metavar="SECONDS",
                    help="after each lost game, ask CP-SAT (with this time limit) whether any word tuple "
                         "other than the true one fits all feedback")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", action="store_true",
                    help="append to --out, skipping games already in it (crashed games are retried)")
    ap.add_argument("--summarize", nargs="+", type=Path, help="only summarize these result files")
    args = ap.parse_args()

    if args.summarize:
        rows = [json.loads(line) for p in args.summarize for line in p.read_text().splitlines() if line]
        print(summarize(rows))
        return

    cfg = {"agents": args.agents, "cipher": args.cipher, "seed_base": args.seed_base, "split": args.split,
           "split_seed": args.split_seed, "zipf": args.zipf, "agent_kwargs": json.loads(args.agent_kwargs),
           "identifiability": args.identifiability, "budget_extra": args.budget_extra,
           "record_decisions": args.record_decisions}
    out = args.out or ROOT / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.cipher}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Interleave agents so partial runs stay paired.
    jobs = [(a, n, g) for n in args.n for g in range(args.games) for a in args.agents]
    rows = []
    mode = "w"
    if args.resume and out.exists():
        old = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        good = [r for r in old if "error" not in r]
        done = {(r["agent"], r["n"], r["game"]) for r in good}
        jobs = [j for j in jobs if j not in done]
        rows = good
        out.write_text("".join(json.dumps(r) + "\n" for r in good))  # drop old error rows
        mode = "a"
        print(f"resuming: {len(done)} games already done, {len(jobs)} to play", flush=True)
    with open(out, mode) as f:
        if args.workers > 1:
            with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
                for r in pool.imap_unordered(play, jobs):
                    f.write(json.dumps(r) + "\n"); f.flush(); rows.append(r)
        else:
            _init(cfg)
            for job in jobs:
                r = play(job)
                f.write(json.dumps(r) + "\n"); f.flush(); rows.append(r)
    print(summarize(rows))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
