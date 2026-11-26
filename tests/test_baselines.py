import math
from collections import Counter

import numpy as np
import pytest

from mystic_wordle import ALPHABET, Cipher, Dictionary, TargetSampler, WordleEnv, score
from mystic_wordle.baselines import ConsistentGuesser, ParticleSolver, RandomGuesser
from mystic_wordle.baselines import kernels as K
from mystic_wordle.baselines.search import CipherSearch, WordTable, decoder_to_cipher


def sample_dict(per_length, seed=0):
    full = Dictionary.load()
    rng = np.random.default_rng(seed)
    words = []
    for L in range(5, 11):
        group = [w for w in full.targets if len(w) == L]
        words += list(rng.choice(group, per_length, replace=False))
    return Dictionary.from_words(words)


@pytest.fixture(scope="module")
def small_dict():
    """20 common words per length, so searches finish fast."""
    return sample_dict(20)


@pytest.fixture(scope="module")
def tiny_dict():
    """6 words per length: small enough that the cipher game is routinely solvable."""
    return sample_dict(6, seed=1)


def decode(d, word):
    return "".join(ALPHABET[d[ord(ch) - 97]] for ch in word)


def played_state(dictionary, seed, n, n_guesses, cipher=None):
    """Play random valid guesses; return (env, obs, CipherSearch synced to obs)."""
    env = WordleEnv(dictionary)
    obs, _ = env.reset(seed, n=n, cipher=cipher)
    rng = np.random.default_rng(seed)
    words = sorted(dictionary.guesses)
    for _ in range(n_guesses):
        obs, *_ = env.step(words[rng.integers(len(words))])
    table = WordTable(sorted(dictionary.targets, key=lambda w: (len(w), w)))
    s = CipherSearch(table, obs, 4, np.random.default_rng(seed))
    s.sync(obs)
    s._refresh_all()
    return env, obs, s


def reference_stats(d, obs, table, s):
    """Python reference for target_stats, via score() on decoded guesses."""
    v, nz = [], []
    for i in range(obs["n"]):
        viols = []
        for w in range(s.lo[i], s.hi[i]):
            word = table.words[w]
            tot = 0
            for guess, fb in obs["history"]:
                if fb[i] is None:
                    continue
                f = score(decode(d, guess), word)
                tot += abs(f.green - fb[i].green) + abs(f.green + f.yellow - fb[i].green - fb[i].yellow)
            viols.append(tot)
        v.append(min(viols))
        nz.append(viols.count(0))
    return v, nz


# -- kernels vs. reference ------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_target_stats_matches_reference(small_dict, seed):
    env, obs, s = played_state(small_dict, seed, n=4, n_guesses=5)
    rng = np.random.default_rng(seed + 100)
    for d in [rng.permutation(26) for _ in range(5)]:
        d = d.astype(np.int64)
        v = np.zeros(s.N, np.int64)
        nz = np.zeros(s.N, np.int64)
        K.target_stats(d, *s._args(), v, nz)
        assert (v.tolist(), nz.tolist()) == reference_stats(d, obs, s.table, s)


def test_true_decoder_is_consistent(small_dict):
    env, obs, s = played_state(small_dict, 7, n=6, n_guesses=8)
    d = (np.frombuffer(env.state.cipher.inverse().mapping.encode(), dtype=np.uint8) - 97).astype(np.int64)
    v = np.zeros(s.N, np.int64)
    nz = np.zeros(s.N, np.int64)
    K.target_stats(d, *s._args(), v, nz)
    assert (v == 0).all() and (nz >= 1).all()
    for i, t in enumerate(env.state.targets):
        assert s.table.index[t] in K.best_words(d, *s._args(), i, 0)


def test_feedback_entropy_matches_reference():
    rng = np.random.default_rng(0)
    words = ["stone", "crane", "planet", "monkeys", "sheets", "llama", "abide", "gardens"]
    S, U = 12, 2
    hyp = [[words[rng.integers(len(words))] for _ in range(U)] for _ in range(S)]
    hw = rng.random(S)
    hw /= hw.sum()
    gt = WordTable(sorted(words, key=lambda w: (len(w), w)))
    h = WordTable([w for row in hyp for w in row])
    h_pos = h.pos.reshape(S, U, -1)
    h_len = h.lens.reshape(S, U)
    h_cnt = h.cnt.reshape(S, U, 26)
    H = K.feedback_entropy(gt.pos, gt.lens, gt.dl, gt.dc, gt.nd, h_pos, h_len, h_cnt, hw)
    for gi, g in enumerate(gt.words):
        mass = Counter()
        for row, w in zip(hyp, hw):
            mass[tuple(score(g, t) for t in row)] += w
        ref = -sum(p * math.log2(p) for p in mass.values())
        assert H[gi] == pytest.approx(ref)


def test_fit_decoder_recovers_cipher_from_true_words(small_dict):
    env, obs, s = played_state(small_dict, 3, n=5, n_guesses=10)
    t = np.array([s.table.index[w] for w in env.state.targets])
    K.seed_rng(0)
    d = np.random.default_rng(0).permutation(26).astype(np.int64)
    assert K.fit_decoder(d, t, s.active, 200_000, 2.0, 200_000, *s._args()[:-2]) == 0


def test_tuple_moves_reach_consistency(small_dict):
    # From random particles with a few guesses of evidence. (With many guesses
    # and no warm start, single-word moves can stall: see README §10.)
    env, obs, s = played_state(small_dict, 5, n=4, n_guesses=3)
    K.seed_rng(1)
    s.move(200, stop_at_zero=True)
    assert s.consistent().all()
    for p in range(len(s.D)):  # the working words are consistent with the decoder
        assert K._tuple_violation(s.D[p], s.T[p], *s._args()[:-2]) == 0


def test_decoder_to_cipher_roundtrip():
    c = Cipher("".join(np.random.default_rng(0).permutation(list(ALPHABET))))
    d = np.frombuffer(c.inverse().mapping.encode(), dtype=np.uint8) - 97
    assert decoder_to_cipher(d) == c


def test_word_table_append_keeps_ranges():
    t = WordTable(["crane", "stone", "planet"])
    before = dict(t.ranges)
    idx = t.append("zzzzzz")
    assert t.words[idx] == "zzzzzz" and t.ranges == before
    assert t.append("stone") == 1
    assert t.pos.shape[0] == t.cnt.shape[0] == t.dl.shape[0] == 4


# -- agents ---------------------------------------------------------------------

def run(agent, env, seed, n, cipher=None):
    obs, _ = env.reset(seed, n=n, cipher=cipher)
    agent.reset(obs, seed)
    guesses, done = [], False
    while not done:
        g = agent.act(obs)
        guesses.append(g)
        obs, _, done, _, info = env.step(g)
        assert not info["rejected"]
    return env.state.won, guesses


def test_random_guesser_is_valid_and_never_repeats(small_dict):
    env = WordleEnv(small_dict)
    won, guesses = run(RandomGuesser(small_dict), env, 0, 8)
    assert len(guesses) == len(set(guesses)) == 8 + 16


@pytest.mark.parametrize("cls", [ConsistentGuesser, ParticleSolver])
def test_search_agents_win_with_known_identity_cipher(small_dict, cls):
    env = WordleEnv(small_dict)
    agent = cls(small_dict, known_cipher=Cipher.identity())
    for seed in range(5):
        won, guesses = run(agent, env, seed, 6, cipher=Cipher.identity())
        assert won
        assert len(guesses) == len(set(guesses))


def test_particle_solver_is_deterministic(small_dict):
    env = WordleEnv(small_dict)
    agent = ParticleSolver(small_dict, n_particles=4, moves=3, rescue_moves=20)
    assert run(agent, env, 11, 4) == run(agent, env, 11, 4)


def test_particle_solver_wins_tiny_cipher_games(tiny_dict):
    """With 6 words per length the cipher game is easy; the solver should win most."""
    env = WordleEnv(tiny_dict)
    agent = ParticleSolver(tiny_dict, n_particles=8)
    wins = sum(run(agent, env, seed, 4)[0] for seed in range(4))
    assert wins >= 3


def test_cipher_estimate_is_a_cipher(small_dict):
    env = WordleEnv(small_dict)
    agent = ConsistentGuesser(small_dict)
    obs, _ = env.reset(0, n=4)
    agent.reset(obs, 0)
    for _ in range(3):
        obs, *_ = env.step(agent.act(obs))
    assert isinstance(agent.cipher_estimate(), Cipher)
