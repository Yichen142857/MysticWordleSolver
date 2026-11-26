import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mystic_wordle import Dictionary, TargetSampler
from mystic_wordle.dictionary import LENGTHS, is_well_formed

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def d():
    return Dictionary.load()


# -- data files ---------------------------------------------------------------

def test_master_file_is_clean(d):
    lines = (DATA / "words.tsv").read_text().splitlines()[1:]
    words = [ln.split("\t")[0] for ln in lines]
    assert len(words) == len(set(words)), "duplicates in words.tsv"
    assert all(is_well_formed(w) for w in words)
    assert all(int(ln.split("\t")[1]) == len(ln.split("\t")[0]) for ln in lines)
    assert len(d.guesses) == len(words)


def test_per_length_files_match_master(d):
    for n in LENGTHS:
        group = (DATA / "by_length" / f"len{n:02d}.txt").read_text().split()
        assert all(len(w) == n for w in group)
        assert set(group) == {w for w in d.guesses if len(w) == n}


def test_stats_match(d):
    stats = json.loads((DATA / "stats.json").read_text())
    assert stats["total_words"] == len(d.guesses)
    assert stats["words_per_length"] == {str(n): c for n, c in d.counts_by_length(d.guesses).items()}


def test_every_length_has_targets(d):
    counts = d.counts_by_length(d.targets)
    assert all(counts[n] >= 100 for n in LENGTHS), counts
    assert set(d.targets) <= d.guesses


def test_target_threshold_is_monotone():
    loose, strict = Dictionary.load(zipf_threshold=2.5), Dictionary.load(zipf_threshold=4.0)
    assert set(strict.targets) < set(loose.targets)
    assert strict.guesses == loose.guesses


def test_rejects_malformed_words():
    with pytest.raises(ValueError):
        Dictionary.from_words(["abcd"])          # too short
    with pytest.raises(ValueError):
        Dictionary.from_words(["Hello"])         # uppercase
    with pytest.raises(ValueError):
        Dictionary.from_words(["hello"], ["world"])  # target not a guess


# -- splits ---------------------------------------------------------------------

def test_split_is_partition_and_deterministic(d):
    s1, s2 = d.split(seed=0), d.split(seed=0)
    assert s1 == s2
    all_words = s1["train"] + s1["val"] + s1["test"]
    assert len(all_words) == len(set(all_words)) == len(d.targets)
    assert d.split(seed=1)["test"] != s1["test"]


def test_split_is_stratified(d):
    s = d.split((0.8, 0.1, 0.1), seed=0)
    total = d.counts_by_length(d.targets)
    test = d.counts_by_length(s["test"])
    for n in LENGTHS:
        assert abs(test[n] - 0.1 * total[n]) <= 1


# -- target sampling ------------------------------------------------------------

def test_targets_distinct_for_all_n(d):
    sampler = TargetSampler(d.split(seed=0)["train"])
    rng = np.random.default_rng(0)
    for n in range(4, 9):
        for _ in range(300):
            t = sampler.sample(n, rng)
            assert len(t) == n == len(set(t))


def test_default_weights_balance_lengths(d):
    sampler = TargetSampler(d.targets)
    rng = np.random.default_rng(0)
    lengths = Counter(len(w) for _ in range(4000) for w in sampler.sample(1, rng))
    for n in LENGTHS:
        assert abs(lengths[n] / 4000 - 1 / 6) < 0.03, lengths


def test_custom_length_weights():
    pool = ["aaaaa", "bbbbb", "cccccc"]
    s = TargetSampler(pool, {5: 0.0, 6: 1.0})
    assert s.sample(1, np.random.default_rng(0)) == ("cccccc",)
    with pytest.raises(ValueError):
        s.sample(2, np.random.default_rng(0))  # only one word has mass


# -- frozen data ----------------------------------------------------------------

def test_frozen_pools_match_the_computed_split(d):
    computed = d.split(seed=0)
    for name in ("train", "val", "test"):
        assert d.target_pool(name) == computed[name]
        assert (DATA / "splits" / f"{name}.txt").read_text().split() == list(computed[name])


def test_manifest_checksums_and_coverage():
    import subprocess
    import sys
    root = DATA.parent
    r = subprocess.run([sys.executable, str(root / "scripts" / "freeze_data.py"), "--check"],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
    man = json.loads((DATA / "manifest.json").read_text())
    assert man["settings"]["target_zipf_threshold"] == 3.0
    assert man["game_rules"]["budget"] == "N + 16"
    assert sum(man["counts"][f"{k}_pool"]["total"] for k in ("train", "val", "test")) == man["counts"]["target_list"]["total"]


def test_non_final_settings_are_computed_not_read():
    loose = Dictionary.load(zipf_threshold=2.5)
    pool = loose.target_pool("test")
    assert pool == loose.split(seed=0)["test"]
    assert len(pool) > len(Dictionary.load().target_pool("test"))
    with pytest.raises(ValueError):
        loose.target_pool("holdout")
