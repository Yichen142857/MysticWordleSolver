"""Build the MysticWordleSolver dictionary files.

Source words: data/raw_data.txt, the ENABLE word list (public domain; no proper nouns,
abbreviations, hyphens or apostrophes). Frequencies: the `wordfreq` package
(Zipf scale: log10 of occurrences per billion words; ~7 is "the", ~1 is very rare).

Outputs (in data/):
  words.tsv               word<TAB>length<TAB>zipf, sorted by length then word.
                          This is the master file the game loads.
  by_length/len05.txt ..  one word per line, per length group (5..10).
  stats.json              counts per length, plus target-list counts at
                          several candidate Zipf thresholds.

The target list is NOT a separate file: it is derived at load time as
words with zipf >= threshold (threshold is an open item, see README §5).

Usage:
  python scripts/build_dictionary.py [--source data/raw_data.txt] [--out data]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from wordfreq import zipf_frequency

MIN_LEN, MAX_LEN = 5, 10
VALID = re.compile(r"^[a-z]+$")
CANDIDATE_THRESHOLDS = [2.5, 3.0, 3.5, 4.0]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=root / "data/raw_data.txt")
    ap.add_argument("--out", type=Path, default=root / "data")
    args = ap.parse_args()

    raw = args.source.read_text().split()
    words = sorted(
        {w.strip().lower() for w in raw if VALID.match(w.strip().lower())
         and MIN_LEN <= len(w.strip()) <= MAX_LEN},
        key=lambda w: (len(w), w),
    )
    rows = [(w, len(w), zipf_frequency(w, "en")) for w in words]

    out = args.out
    (out / "by_length").mkdir(parents=True, exist_ok=True)
    with open(out / "words.tsv", "w") as f:
        f.write("word\tlength\tzipf\n")
        for w, n, z in rows:
            f.write(f"{w}\t{n}\t{z:.2f}\n")
    for n in range(MIN_LEN, MAX_LEN + 1):
        group = [w for w, L, _ in rows if L == n]
        (out / "by_length" / f"len{n:02d}.txt").write_text("\n".join(group) + "\n")

    by_len = Counter(L for _, L, _ in rows)
    stats = {
        "source": args.source.name,
        "total_words": len(rows),
        "words_per_length": {str(n): by_len[n] for n in range(MIN_LEN, MAX_LEN + 1)},
        "targets_per_length_at_zipf_threshold": {
            str(t): {str(n): sum(1 for _, L, z in rows if L == n and z >= t)
                     for n in range(MIN_LEN, MAX_LEN + 1)}
            for t in CANDIDATE_THRESHOLDS
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
