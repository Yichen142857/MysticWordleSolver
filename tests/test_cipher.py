import numpy as np
import pytest

from mystic_wordle import ALPHABET, Cipher, CipherSampler


def test_sampled_ciphers_are_bijections():
    s = CipherSampler("random")
    rng = np.random.default_rng(0)
    for _ in range(500):
        c = s.sample(rng)
        assert sorted(c.mapping) == list(ALPHABET)
        assert c.decode(c.encode(ALPHABET)) == ALPHABET
        assert c.inverse().encode(c.encode(ALPHABET)) == ALPHABET


def test_random_ciphers_vary_and_are_not_all_identity():
    rng = np.random.default_rng(0)
    seen = {CipherSampler("random").sample(rng) for _ in range(50)}
    assert len(seen) == 50
    assert Cipher.identity() not in seen


def test_invalid_mapping_rejected():
    with pytest.raises(ValueError):
        Cipher("a" * 26)
    with pytest.raises(ValueError):
        Cipher(ALPHABET[:-1])


def test_modes():
    rng = np.random.default_rng(0)
    assert CipherSampler("identity").sample(rng) == Cipher.identity()
    fixed = CipherSampler("fixed", fixed_seed=3)
    assert fixed.sample(rng) == fixed.sample(np.random.default_rng(99))
    assert fixed.fixed != Cipher.identity()
    with pytest.raises(ValueError):
        CipherSampler("nope")


def test_from_swaps():
    c = Cipher.from_swaps("te", "ab")
    assert c.encode("stoneab") == "seontba"
    assert c.inverse() == c  # product of disjoint transpositions is an involution
