import numpy as np
import pytest

from mystic_wordle import Cipher, CipherSampler, Dictionary, Rewards, TargetSampler, WordleEnv, score

WORDS = ["stone", "crane", "sheets", "abide", "llama", "hello", "planet",
         "monkeys", "gardens", "elephant", "mountains", "basketball", "jumpy", "fjord"]
TARGETS = ["stone", "crane", "planet", "monkeys", "gardens", "elephant", "mountains", "basketball"]
R = Rewards()


@pytest.fixture
def small():
    return Dictionary.from_words(WORDS, TARGETS)


@pytest.fixture
def env(small):
    return WordleEnv(small)


def start(env, targets=("stone", "crane", "planet", "monkeys"), cipher=None):
    obs, _ = env.reset(0, targets=targets, cipher=cipher or Cipher.from_swaps("te"))
    return obs


# -- observation ----------------------------------------------------------------

def test_initial_observation(env):
    obs = start(env)
    assert obs == {
        "n": 4, "target_lengths": (5, 5, 6, 7), "budget_total": 20, "budget_left": 20,
        "rejections_left": 20, "solved": (False,) * 4, "history": (),
    }


def test_observation_hides_cipher_and_targets(env):
    obs = start(env)
    flat = repr(obs)
    for t in env.state.targets + env.state.ciphered:
        assert t not in flat
    assert env.state.cipher.mapping not in flat


# -- feedback and solving -------------------------------------------------------

def test_feedback_is_against_ciphered_targets(env):
    start(env)
    obs, reward, term, trunc, info = env.step("sheets")
    _, fb = obs["history"][-1]
    assert fb[0] == (2, 1)  # README §3 example
    assert fb == tuple(score("sheets", c) for c in env.state.ciphered)
    assert reward == pytest.approx(R.valid_guess)
    assert not term and not trunc and info["solved_now"] is None


def test_plaintext_match_solves_and_stops_feedback(env):
    start(env)
    obs, reward, *_ , info = env.step("stone")
    assert info["solved_now"] == 0 and obs["solved"] == (True, False, False, False)
    assert reward == pytest.approx(R.solve + R.valid_guess)
    # The solving guess still returns its feedback pair, scored against c(stone).
    assert obs["history"][-1][1][0] == score("stone", "seont") == (3, 2)
    obs, reward, *_ , info = env.step("stone")  # repeat: no second reward, no feedback
    assert obs["history"][-1][1][0] is None
    assert info["solved_now"] is None and reward == pytest.approx(R.valid_guess)
    obs, *_ = env.step("crane")
    assert obs["history"][-1][1][0] is None and obs["solved"][:2] == (True, True)


def test_win_rewards_and_termination(env):
    start(env, targets=("stone", "crane", "planet", "monkeys"))
    total = 0.0
    for w in ["stone", "crane", "planet"]:
        _, r, term, *_ = env.step(w)
        total += r
        assert not term
    _, r, term, _, info = env.step("monkeys")
    total += r
    assert term and info["won"] and env.state.won
    assert total == pytest.approx(4 * (R.solve + R.valid_guess) + R.win_bonus)
    with pytest.raises(RuntimeError):
        env.step("stone")


def test_solve_on_last_guess_is_win(env):
    start(env)
    for w in ["stone", "crane", "planet"]:
        env.step(w)
    for _ in range(env.state.budget_left - 1):
        env.step("jumpy")
    assert env.state.budget_left == 1
    _, r, term, _, info = env.step("monkeys")
    assert term and info["won"] and env.state.budget_left == 0
    assert r == pytest.approx(R.solve + R.valid_guess + R.win_bonus)


def test_budget_runs_out_loss(env):
    obs = start(env)
    assert obs["budget_left"] == obs["n"] + 16
    for i in range(obs["budget_total"]):
        obs, _, term, *_ = env.step("jumpy")
        assert term == (i == obs["budget_total"] - 1)
    assert obs["budget_left"] == 0 and not env.state.won and not any(obs["solved"])


def test_repeated_guess_costs_budget_and_repeats_feedback(env):
    start(env)
    a, *_ = env.step("llama")
    b, *_ = env.step("llama")
    assert b["budget_left"] == a["budget_left"] - 1
    assert b["history"][-1] == a["history"][-1]


def test_identity_cipher_solving_guess_is_all_green(env):
    start(env, cipher=Cipher.identity())
    obs, *_ = env.step("stone")
    assert obs["history"][-1][1][0] == (5, 0)


# -- rejections -----------------------------------------------------------------

@pytest.mark.parametrize("bad,reason", [
    ("stones", "not in dictionary"),
    ("abcd", "length"),
    ("abcdefghijk", "length"),
    ("Stone", "lowercase"),
    ("st0ne", "lowercase"),
    (12345, "not a string"),
])
def test_rejected_guess_changes_nothing_but_rejection_count(env, bad, reason):
    before = start(env)
    obs, r, term, trunc, info = env.step(bad)
    assert info["rejected"] and reason in info["reason"]
    assert r == pytest.approx(R.rejected_guess) and not term and not trunc
    assert obs["rejections_left"] == before["rejections_left"] - 1
    assert {k: v for k, v in obs.items() if k != "rejections_left"} == \
           {k: v for k, v in before.items() if k != "rejections_left"}
    assert env.state.rejection_log == [(bad, info["reason"])]


def test_rejection_cap_ends_game(small):
    env = WordleEnv(small, max_rejections=3)
    start(env)
    env.step("stone")
    for i in range(3):
        _, r, term, *_ = env.step("zzzzz")
        assert term == (i == 2)
    assert env.state.done and not env.state.won
    with pytest.raises(RuntimeError):
        env.step("crane")


# -- sampling and seeding -------------------------------------------------------

def test_sampled_games_have_distinct_targets_and_n_in_range(small):
    env = WordleEnv(small)
    ns = set()
    for seed in range(300):
        obs, _ = env.reset(seed)
        t = env.state.targets
        assert 4 <= obs["n"] <= 8 and len(set(t)) == len(t) == obs["n"]
        assert set(t) <= set(TARGETS)
        assert obs["budget_total"] == obs["n"] + 16
        ns.add(obs["n"])
    assert ns == {4, 5, 6, 7, 8}


def test_reset_is_deterministic(env):
    env.reset(42)
    a = (env.state.targets, env.state.cipher)
    env.reset(7)
    env.reset(42)
    assert (env.state.targets, env.state.cipher) == a
    env.reset(43)
    assert (env.state.targets, env.state.cipher) != a


def test_same_seed_same_targets_across_cipher_modes(small):
    rand = WordleEnv(small, cipher_sampler=CipherSampler("random"))
    ident = WordleEnv(small, cipher_sampler=CipherSampler("identity"))
    for seed in range(20):
        rand.reset(seed)
        ident.reset(seed)
        assert rand.state.targets == ident.state.targets
        assert ident.state.cipher == Cipher.identity()


def test_step_before_reset_raises(small):
    with pytest.raises(RuntimeError):
        WordleEnv(small).step("stone")


def test_full_dictionary_smoke():
    d = Dictionary.load()
    test_pool = d.split(seed=0)["test"]
    env = WordleEnv(d, target_sampler=TargetSampler(test_pool))
    rng = np.random.default_rng(0)
    guesses = sorted(d.guesses)
    for seed in range(50):
        obs, _ = env.reset(seed)
        assert set(env.state.targets) <= set(test_pool)
        term = False
        while not term:
            obs, _, term, *_ = env.step(guesses[rng.integers(len(guesses))])
        assert obs["budget_left"] == 0 or env.state.won
