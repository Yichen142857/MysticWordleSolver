"""Post-hoc identifiability checks on saved evaluation results.

For each game in the given results files, replay its guesses and ask CP-SAT
whether a word tuple other than the true one fits all feedback after the first
k guesses, for every k in --at (default: B - N and the end of the game).

  python scripts/identifiability.py results/pilot-exact.jsonl --at 6 8 10 end --seconds 60 --out ident.jsonl
  python scripts/identifiability.py results/pilot-exact.jsonl --at 6 8 10 end --out ident.jsonl --resume
  python scripts/identifiability.py --summarize ident.jsonl

The summary gives, per (agent, N), the first checked guess count at which the
targets were unique in each game, and the share of games unique at each k.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mystic_wordle import Dictionary, TargetSampler, WordleEnv  # noqa: E402
from mystic_wordle.baselines.exact import alternative_exists  # noqa: E402
from mystic_wordle.baselines.search import WordTable  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import replay  # noqa: E402


def summarize(paths):
    """Per (agent, N): first checkpoint at which each game was unique, and the share
    unique at each fixed checkpoint. The end-of-game check is reported separately.

    Older output files stored the end check as k=<game length>, always as the
    row's last key and with no "end" key. For such rows the last k is read as
    the end check and left out of the checkpoint table.
    """
    from collections import defaultdict
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.startswith("{")]
    groups = defaultdict(list)
    for r in rows:
        groups[r["agent"], r["n"]].append(r)
    for (agent, n), rs in sorted(groups.items()):
        checks, firsts, ends = [], [], []
        for r in rs:
            ks = [k for k in r if k.startswith("k=")]
            end = r.get("end")
            end_k = r.get("end_k")
            if end is None and ks:  # old format: last k is the end check
                end, end_k = r[ks[-1]], int(ks[-1][2:])
                ks = ks[:-1]
            vals = {int(k[2:]): r[k] for k in ks}
            checks.append(vals)
            cands = [k for k in vals if vals[k] == "unique"]
            if end == "unique" and end_k is not None:
                cands.append(end_k)
            first = min(cands) if cands else ("end" if end == "unique" else None)
            firsts.append("never" if first is None else first)
            ends.append(end)
        won = sum(r.get("won", False) for r in rs)
        print(f"{agent} N={n}: {len(rs)} games ({won} won)")
        print("  first k unique per game:", firsts)
        for k in sorted({k for v in checks for k in v}):
            vals = [v[k] for v in checks if k in v]
            u, a, t = vals.count("unique"), vals.count("ambiguous"), vals.count("timeout")
            print(f"  k={k:2d}: unique {u}/{u + a}" + (f" (+{t} timeout)" if t else ""))
        u, a, t = ends.count("unique"), ends.count("ambiguous"), ends.count("timeout")
        print(f"  end : unique {u}/{u + a}" + (f" (+{t} timeout)" if t else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--summarize", nargs="+", type=Path, help="summarize saved output of this script")
    ap.add_argument("--at", nargs="+", default=["probe", "end"], help="guess counts, 'probe' (= B - N) or 'end'")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--zipf", type=float, default=3.0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--budget-extra", type=int, default=None,
                    help="budget used in the run (default: read from each row)")
    ap.add_argument("--out", type=Path, help="write results here (default: stdout)")
    ap.add_argument("--resume", action="store_true", help="with --out: skip games already in it")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return

    d = Dictionary.load(zipf_threshold=args.zipf)
    pool = TargetSampler(d.target_pool(args.split, seed=args.split_seed))
    table = WordTable(d.targets)
    done = set()
    if args.out and args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                done.add((r["agent"], r["n"], r["game"]))
        print(f"resuming: {len(done)} games already checked", file=sys.stderr)
    sink = open(args.out, "a" if args.resume else "w") if args.out else sys.stdout
    for path in args.files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                print(f"skip {r['agent']} n={r['n']} game={r['game']}: game crashed", file=sys.stderr)
                continue
            if "guess_list" not in r:
                print(f"skip {r['agent']} n={r['n']} game={r['game']}: no guess_list saved", file=sys.stderr)
                continue
            if (r["agent"], r["n"], r["game"]) in done:
                continue
            extra = args.budget_extra if args.budget_extra is not None else r["budget"] - r["n"]
            env = WordleEnv(d, pool, budget_fn=lambda n, e=extra: n + e)
            res = {}
            for at in args.at:
                k = {"probe": r["budget"] - r["n"], "end": len(r["guess_list"])}.get(at) or int(at)
                obs = replay(env, r["seed"], r["n"], r["guess_list"][:k])
                alt = alternative_exists(table, obs, env.state.targets, args.seconds)
                key = "end" if at == "end" else f"k={k}"
                res[key] = {True: "ambiguous", False: "unique", None: "timeout"}[alt]
                if at == "end":
                    res["end_k"] = k
            print(json.dumps({"agent": r["agent"], "n": r["n"], "game": r["game"], "won": r["won"], **res}),
                  file=sink, flush=True)


if __name__ == "__main__":
    main()
