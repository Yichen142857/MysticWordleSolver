import random
from collections import Counter

import pytest

from mystic_wordle import ALPHABET, Cipher, score


def standard_wordle_colors(guess: str, target: str) -> str:
    """Independent reference: standard Wordle per-position coloring (equal lengths).
    G = green, Y = yellow, . = gray."""
    assert len(guess) == len(target)
    colors = ["."] * len(guess)
    remaining = Counter()
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            colors[i] = "G"
        else:
            remaining[t] += 1
    for i, g in enumerate(guess):
        if colors[i] != "G" and remaining[g] > 0:
            colors[i] = "Y"
            remaining[g] -= 1
    return "".join(colors)


# Hand-checked (guess, target, standard colors, expected counts).
EQUAL_LENGTH_CASES = [
    ("crane", "crane", "GGGGG", (5, 0)),
    ("speed", "abide", "..Y.Y", (0, 2)),   # second e gray: target has one e
    ("eerie", "elder", "GYY..", (1, 2)),   # green e, yellow e, yellow r; last e gray
    ("llama", "hello", "YY...", (0, 2)),   # both l's yellow, a's gray
    ("geese", "eerie", ".GY.G", (2, 1)),   # repeated letters on both sides
    ("jumpy", "strai", ".....", (0, 0)),
]


@pytest.mark.parametrize("guess,target,colors,expected", EQUAL_LENGTH_CASES)
def test_identity_matches_standard_wordle(guess, target, colors, expected):
    assert standard_wordle_colors(guess, target) == colors
    ident = Cipher.identity()
    assert score(guess, ident.encode(target)) == expected
    assert expected == (colors.count("G"), colors.count("Y"))


# Hand-computed unequal-length cases (standard Wordle has no reference here).
UNEQUAL_LENGTH_CASES = [
    ("sheets", "seont", (2, 1)),     # README §3 example
    ("sheets", "stone", (1, 2)),     # same, without the cipher
    ("abcdefg", "gfedc", (1, 4)),    # green d at pos 4; tail letters can be yellow
    ("stones", "stone", (5, 0)),     # extra guess letter s has nothing left to match
    ("stone", "stones", (5, 0)),
    ("aaaaaa", "banana", (3, 0)),    # greens at 2,4,6 use up every a in the target
]


@pytest.mark.parametrize("guess,target,expected", UNEQUAL_LENGTH_CASES)
def test_unequal_lengths_hand_checked(guess, target, expected):
    assert score(guess, target) == expected


def test_readme_example_cipher_changes_feedback():
    c = Cipher.from_swaps("te")
    assert c.encode("stone") == "seont"
    assert score("sheets", c.encode("stone")) == (2, 1)
    assert score("sheets", "stone") == (1, 2)


def _rand_word(rng, lo=5, hi=10, alphabet=ALPHABET):
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))


def test_property_total_is_multiset_intersection():
    rng = random.Random(0)
    for _ in range(5000):
        g = _rand_word(rng, alphabet="abcdef")  # small alphabet -> many repeats
        t = _rand_word(rng, alphabet="abcdef")
        fb = score(g, t)
        assert fb.green + fb.yellow == sum((Counter(g) & Counter(t)).values())
        assert fb.green == sum(a == b for a, b in zip(g, t))
        assert 0 <= fb.green <= min(len(g), len(t))
        assert fb.green + fb.yellow <= min(len(g), len(t))


def test_property_cipher_equivalence():
    """score(g, c(t)) == score(c^-1(g), t): the cipher can be moved onto the guess."""
    rng = random.Random(1)
    for _ in range(2000):
        perm = list(ALPHABET)
        rng.shuffle(perm)
        c = Cipher("".join(perm))
        g, t = _rand_word(rng), _rand_word(rng)
        assert score(g, c.encode(t)) == score(c.decode(g), t)


def test_property_matches_standard_wordle_on_random_equal_lengths():
    rng = random.Random(2)
    for _ in range(3000):
        n = rng.randint(5, 10)
        g = _rand_word(rng, n, n, "abcde")
        t = _rand_word(rng, n, n, "abcde")
        colors = standard_wordle_colors(g, t)
        assert score(g, t) == (colors.count("G"), colors.count("Y"))
