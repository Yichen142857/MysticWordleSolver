import numpy as np
import pytest

from mystic_wordle import Cipher, WordleEnv
from mystic_wordle.baselines import kernels as K
from mystic_wordle.baselines.exact import ExactSolver, GameModel, alternative_exists
from mystic_wordle.baselines.search import CipherSearch, WordTable

from test_baselines import run, sample_dict

pytest.importorskip("ortools")


@pytest.fixture(scope="module")
def tiny_dict():
    return sample_dict(6, seed=1)


@pytest.fixture(scope="module")
def small_dict():
    return sample_dict(20)


def game_state(dictionary, seed, n, n_guesses):
    env = WordleEnv(dictionary)
    obs, _ = env.reset(seed, n=n)
    rng = np.random.default_rng(seed)
    words = sorted(dictionary.guesses)
    for _ in range(n_guesses):
        obs, *_ = env.step(words[rng.integers(len(words))])
    table = WordTable(sorted(dictionary.targets, key=lambda w: (len(w), w)))
    s = CipherSearch(table, obs, 1, np.random.default_rng(seed))
    s.sync(obs)
    return env, obs, table, s


def add_particle(s, d, words):
    s.D = np.vstack([s.D, d[None]])
    s.T = np.vstack([s.T, words[None]])
    s.v = np.vstack([s.v, np.zeros((1, s.N), np.int64)])
    s.nz = np.vstack([s.nz, np.zeros((1, s.N), np.int64)])
    s.logpi = np.append(s.logpi, 0.0)
    s.logw = np.append(s.logw, 0.0)
    s._refresh(len(s.D) - 1)
    return len(s.D) - 1


@pytest.mark.parametrize("seed", range(3))
def test_model_solutions_pass_the_kernel_check(small_dict, seed):
    env, obs, table, s = game_state(small_dict, seed, n=4, n_guesses=6)
    model = GameModel(s, obs["history"])
    status, d, words = model.solve(30, seed)
    assert status == "found"  # the true assignment always exists
    assert K._tuple_violation(d, words, *s._args()[:-2]) == 0


def test_coverage_cuts_enumerate_a_complete_support(tiny_dict):
    """Solve-and-cut until infeasible; the particles must then cover the true tuple
    and every tuple consistent under the true decoder."""
    env, obs, table, s = game_state(tiny_dict, 2, n=4, n_guesses=8)
    s.D, s.T = s.D[:0], s.T[:0]
    s.v, s.nz, s.logpi, s.logw = s.v[:0], s.nz[:0], s.logpi[:0], s.logw[:0]
    model = GameModel(s, obs["history"])
    for _ in range(200):
        status, d, words = model.solve(30, 0)
        if status == "infeasible":
            break
        assert status == "found"
        model.cut(add_particle(s, d, words))
    assert status == "infeasible"
    truth = [table.index[t] for t in env.state.targets]
    covered = lambda tup: any(all(tup[i] in s.candidates(p, i) for i in range(s.N)) for p in range(len(s.D)))
    assert covered(truth)
    true_d = (np.frombuffer(env.state.cipher.inverse().mapping.encode(), dtype=np.uint8) - 97).astype(np.int64)
    sets = [K.best_words(true_d, *s._args(), i, 0) for i in range(s.N)]
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert covered([int(rng.choice(x)) for x in sets])


def test_alternative_exists_early_and_not_after_full_information(tiny_dict):
    env, obs, table, s = game_state(tiny_dict, 3, n=4, n_guesses=1)
    assert alternative_exists(table, obs, env.state.targets, 30) is True
    # Every target solved: nothing else can fit.
    env2 = WordleEnv(tiny_dict)
    obs2, _ = env2.reset(3, n=4)
    for t in env2.state.targets:
        obs2, *_ = env2.step(t)
    assert alternative_exists(table, obs2, env2.state.targets, 30) is False


def test_exact_solver_particles_are_always_consistent(small_dict):
    env = WordleEnv(small_dict)
    agent = ExactSolver(small_dict, cp_seconds=3)
    obs, _ = env.reset(4, n=4)
    agent.reset(obs, 4)
    for _ in range(6):
        obs, _, done, *_ = env.step(agent.act(obs))
        if done:
            break
        agent._update(obs)
        s = agent.search
        if s.consistent().any():
            assert s.consistent().all()


def test_exact_solver_wins_tiny_cipher_games(tiny_dict):
    env = WordleEnv(tiny_dict)
    agent = ExactSolver(tiny_dict, cp_seconds=3)
    assert sum(run(agent, env, seed, 4)[0] for seed in range(4)) >= 3


def test_exact_solver_known_cipher(tiny_dict):
    env = WordleEnv(tiny_dict)
    agent = ExactSolver(tiny_dict, known_cipher=Cipher.identity())
    assert run(agent, env, 0, 6, cipher=Cipher.identity())[0]


def test_solver_exception_retries_without_hints(small_dict, monkeypatch):
    env, obs, table, s = game_state(small_dict, 0, n=4, n_guesses=5)
    model = GameModel(s, obs["history"])
    real = GameModel._run
    calls = []

    def flaky(self, seconds, seed):
        calls.append(seed)
        if len(calls) == 1:
            raise IndexError("absl::btree_map::at")
        return real(self, seconds, seed)

    monkeypatch.setattr(GameModel, "_run", flaky)
    status, d, words = model.solve(30, 0)
    assert status == "found" and len(calls) == 2
    assert model.errors and "IndexError" in model.errors[0]


def test_exact_solver_survives_a_broken_cp_sat(tiny_dict, monkeypatch):
    def broken(self, seconds, seed):
        raise IndexError("absl::btree_map::at")

    monkeypatch.setattr(GameModel, "_run", broken)
    env = WordleEnv(tiny_dict)
    agent = ExactSolver(tiny_dict, cp_seconds=1)
    won, guesses = run(agent, env, 0, 4)  # must finish the game, whatever the result
    assert len(guesses) >= 4
    assert any("solver_errors" in t for t in agent.stats)


def test_decision_records_match_the_guess(tiny_dict):
    env = WordleEnv(tiny_dict)
    agent = ExactSolver(tiny_dict, cp_seconds=2)
    obs, _ = env.reset(1, n=4)
    agent.reset(obs, 1)
    done = False
    while not done:
        agent.last_decision = None
        g = agent.act(obs)
        dec = agent.last_decision
        assert dec["guess"] == g and dec["mode"] in ("probe", "commit", "forced")
        unsolved = [str(i) for i, ok in enumerate(obs["solved"]) if not ok]
        assert sorted(dec["marginals"]) == sorted(unsolved)
        for top in dec["marginals"].values():
            assert all(0 <= p <= 1.0001 for _, p in top)
        obs, _, done, *_ = env.step(g)


def test_evaluate_records_decisions(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "r.jsonl"
    subprocess.run([sys.executable, str(root / "scripts" / "evaluate.py"), "--agents", "consistent",
                    "--cipher", "identity", "--n", "4", "--games", "1", "--record-decisions",
                    "--out", str(out)], check=True, capture_output=True, cwd=root)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["budget"] == 4 + 16
    assert len(row["decisions"]) == len(row["guess_list"])
    assert all(d["guess"] == g for d, g in zip(row["decisions"], row["guess_list"]))
