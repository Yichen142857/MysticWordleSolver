"""Freeze the final target pools and write the data manifest (README §5).

Writes data/splits/{train,val,test}.txt (one word per line, sorted by length
then word) and data/manifest.json (settings, counts per length and SHA-256
checksums). Run once after build_dictionary.py; the files are then the source
of truth for the pools (Dictionary.target_pool reads them).

  python scripts/freeze_data.py            # write (refuses to overwrite changed pools)
  python scripts/freeze_data.py --check    # verify files against words.tsv and the manifest
  python scripts/freeze_data.py --force    # overwrite even if the pools would change
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mystic_wordle.dictionary import (  # noqa: E402
    DATA_DIR, DEFAULT_DATA, DEFAULT_SPLIT, DEFAULT_SPLIT_SEED, DEFAULT_ZIPF_THRESHOLD, LENGTHS, MANIFEST,
    SPLIT_NAMES, SPLITS_DIR, Dictionary)
from mystic_wordle.game import DEFAULT_MAX_REJECTIONS, DEFAULT_N_RANGE, default_budget  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_length(words) -> dict[str, int]:
    out = {str(n): 0 for n in LENGTHS}
    for w in words:
        out[str(len(w))] += 1
    return out


def build_manifest(d: Dictionary, pools: dict) -> dict:
    raw = DATA_DIR / "raw_data.txt"
    return {
        "description": "Frozen MysticWordleSolver data. Pools are read from data/splits/*.txt.",
        "source_word_list": {"file": "data/raw_data.txt", "name": "ENABLE (public domain)",
                             "sha256": sha256(raw) if raw.exists() else None},
        "words_tsv": {"file": "data/words.tsv", "sha256": sha256(DEFAULT_DATA),
                      "frequency_source": "wordfreq Zipf frequency, English"},
        "settings": {
            "word_lengths": [min(LENGTHS), max(LENGTHS)],
            "target_zipf_threshold": DEFAULT_ZIPF_THRESHOLD,
            "split_ratios": dict(zip(SPLIT_NAMES, DEFAULT_SPLIT)),
            "split_seed": DEFAULT_SPLIT_SEED,
            "split_method": "per length group, numpy default_rng(seed).permutation, then train/val/test slices",
            "target_length_weights": "equal probability per length group, uniform within a group "
                                     "(TargetSampler default)",
            "evaluation_pool": "test",
            "tuning_pool": "val",
        },
        "game_rules": {"n_targets": list(DEFAULT_N_RANGE), "budget": f"N + {default_budget(0)}",
                       "max_rejections": DEFAULT_MAX_REJECTIONS},
        "counts": {
            "guess_dictionary": {"total": len(d.guesses), "per_length": by_length(d.guesses)},
            "target_list": {"total": len(d.targets), "per_length": by_length(d.targets)},
            **{f"{k}_pool": {"total": len(v), "per_length": by_length(v),
                             "file": f"data/splits/{k}.txt", "sha256": sha256(SPLITS_DIR / f"{k}.txt")}
               for k, v in pools.items()},
        },
    }


def check() -> list[str]:
    problems = []
    if not MANIFEST.exists():
        return [f"{MANIFEST} missing"]
    man = json.loads(MANIFEST.read_text())
    if sha256(DEFAULT_DATA) != man["words_tsv"]["sha256"]:
        problems.append("data/words.tsv changed since the manifest was written")
    d = Dictionary.load()
    seen = set()
    for k in SPLIT_NAMES:
        f = SPLITS_DIR / f"{k}.txt"
        if not f.exists():
            problems.append(f"{f} missing")
            continue
        if sha256(f) != man["counts"][f"{k}_pool"]["sha256"]:
            problems.append(f"{f} changed since the manifest was written")
        words = f.read_text().split()
        if seen & set(words):
            problems.append(f"{k} pool overlaps another pool")
        seen |= set(words)
    if seen != set(d.targets):
        problems.append("pools do not cover exactly the target list")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.check:
        problems = check()
        print("data OK" if not problems else "\n".join(problems))
        sys.exit(1 if problems else 0)

    d = Dictionary.load()
    pools = d.split(DEFAULT_SPLIT, DEFAULT_SPLIT_SEED)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for k, words in pools.items():
        f = SPLITS_DIR / f"{k}.txt"
        if f.exists() and f.read_text().split() != list(words) and not args.force:
            sys.exit(f"{f} exists and differs from the computed pool; refusing to overwrite (use --force)")
        f.write_text("\n".join(words) + "\n")
    MANIFEST.write_text(json.dumps(build_manifest(d, pools), indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in pools.items()}), f"-> {SPLITS_DIR}, {MANIFEST}")


if __name__ == "__main__":
    main()
