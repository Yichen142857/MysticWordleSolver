# MysticWordleSolver: Cipher Wordle Multi-Target Solver Plan

## 1. Problem

Each game has N hidden target words (N = 4 to 8) and a hidden letter cipher (a random permutation of a–z). The agent guesses plaintext dictionary words. Feedback is computed against the **ciphered** targets, so the agent must infer the cipher and the targets together from count-based feedback.

Equivalent view: since a bijection preserves matches, `score(g, c(t)) = score(c⁻¹(g), t)`. The game is multi-target count-feedback Wordle in which every guess passes through an unknown cipher before scoring. The agent chooses its words, but not the letters it is actually probing.

## 2. Rules

- **Targets:** N distinct words from the target list (section 5), N uniform over {4..8}.
- **Budget:** B(N) = N + 16, set by the budget study (section 10). The exact-constraint solver wins about 84% of games at this budget, about the same for every N.
- **Cipher:** uniform permutation of a–z, one per game, shared by all targets. Applied to **targets only**. Applying it to guesses too would cancel its effect on feedback.
- **Valid guess:** word in the guess dictionary, length 5–10, lowercase a–z. Test-pool words are allowed.
- **Invalid guess:** rejected. No feedback, no budget cost, not in history. Logged.
- **Rejection cap:** after R_max rejections in one game, the game ends as a loss. This guarantees termination, since rejections cost no budget.
- **Repeated guess:** allowed. Costs budget and returns normal feedback.
- **Solve:** guess equals the plaintext target. The target is marked solved, reported explicitly, and no longer scored. The solving guess still returns its feedback pair. It is generally not all green, because feedback is computed against the ciphered target.
- **Win:** all N solved within budget, including a solve on the last guess. **Loss:** budget or rejection cap runs out first.
- **Agent sees:** N, target lengths, budget left, rejections left, solved flags, and the guess history with per-target feedback. **Never sees** the cipher or the targets.

## 3. Feedback

Feedback is a pair of counts per target, (green, yellow), with no per-position colors. For each target, compare guess g with ciphered target c(t):

1. **Green:** positions 1..min(L_g, L_t) where letters match. Mark both used.
2. **Yellow:** multiset-match the unused guess letters against the unused target letters. Each count is one yellow.

Invariant: green + yellow = size of the multiset intersection of g and c(t).

Example: target `stone`, cipher swaps t↔e, so c(`stone`) = `seont`. Guess `sheets` gives **(2 green, 1 yellow)**. Without the cipher it would be (1, 2).

## 4. Reward

| Event | Reward |
|---|---|
| Target newly solved | +1 |
| Win (all N solved) | +2 bonus |
| Valid guess | −0.02 |
| Rejected guess | −0.1 |

Solves dominate: one solve is worth 50 guesses. Rejections are free in budget, so their penalty is larger than a guess's cost. Rejections can only happen for text-generating policies. Index-based policies always pick a valid word. The reward is for training only. Evaluation uses solve metrics.

## 5. Data

- **Guess dictionary:** words of length 5–10, lowercase a–z, no duplicates, no proper nouns or abbreviations. Can be large.
- **Target list:** the subset of the guess dictionary above a word-frequency threshold. Obscure targets flatten the prior and make cryptogram solutions ambiguous.
- Record counts per length group for both lists.
- **Splits:** train, validation, and test pools drawn from the target list. Evaluation targets come only from the test pool. Guesses use the full guess dictionary.
- **Priors:** every method uses the same prior, the full target list with the target weights. No method may use split membership. A prior restricted to the train pool can never solve test games, and one restricted to the test pool leaks.
- **Target weights:** per-word weight by length group, so rare lengths appear often enough. The same formula is used in every split.

**Final settings (frozen; `data/manifest.json` records them with counts and checksums):**

| Setting | Value | Why |
|---|---|---|
| Guess dictionary | all 120,572 ENABLE words of length 5–10 | public domain, no proper nouns or abbreviations |
| Target list | guess words with Zipf frequency ≥ 3.0: 17,724 words, 1,952–3,649 per length | common words; every budget and baseline result uses this list |
| Pools | train / val / test = 80 / 10 / 10 % per length, seed 0: 14,179 / 1,772 / 1,773 words | stratified, so every length is in every pool |
| Target weights | equal probability per length group, uniform within a group | short targets are the hardest (exact solver: 86% at 5 letters vs 97% at 10) and stay well represented |

- **Frozen pools:** the pools are stored as `data/splits/{train,val,test}.txt` and `Dictionary.target_pool(name)` reads them. Changes to numpy or to the split code cannot move words between pools. The frozen files match the split every earlier run used: all 2,700 target draws from the budget runs fall inside the frozen test pool.
- **Pool use:** train for training data and demonstrations, val for every tuning decision from now on, test only for final evaluations. The budget study used the test pool, before this rule was in place. Other settings (`Dictionary.load(zipf_threshold=...)`, other seeds or ratios) are computed on the fly. They change the difficulty, so the budget study would have to be redone for them.
- `python scripts/freeze_data.py --check` verifies the files against `words.tsv` and the manifest.

## 6. Build

- **Components:** `Dictionary`, `CipherSampler`, `TargetSampler`, `FeedbackScorer` (pure function `score(guess, ciphered_target)`), `GameState`, and `WordleEnv` (`reset(seed)`, `step(guess)`, Gym-style).
- **Practices:** seedable RNG. `step` validates the guess first. `Dictionary` is loaded from a file, so a draft word list can be swapped for the final one without code changes.
- **Tests:**
  - Cipher is a bijection.
  - Identity cipher matches green/yellow counts derived from standard Wordle coloring on hand-checked equal-length cases, including repeated letters.
  - Unequal-length cases match hand-computed counts.
  - green + yellow equals the multiset intersection size (property test).
  - `score(g, c(t)) == score(c⁻¹(g), t)` for random ciphers (property test).
  - A non-identity cipher changes feedback in at least one case.
  - Plaintext match marks solved, and solved targets get no later feedback.
  - Solving the last target on the last guess is a win.
  - Targets are distinct for N = 4..8.
  - Rejected guesses change nothing except the rejection count. The game ends at R_max.
  - Budget starts at B(N) and ends at 0.
  - `reset(seed)` is deterministic.

## 7. Plan

1. **Spec:** resolve the open items in section 9 that block code (N range, one cipher per game, R_max).
2. **Core logic:** `FeedbackScorer`, `CipherSampler`, `GameState`, and `WordleEnv` against a draft word list. Pass the tests.
3. **Reference solver and solvability study:** build the particle solver below and run it for N = 4..8, with and without the cipher, over a range of budgets. Set B(N) as the smallest budget at which the solver reaches the target solve rate. The result is a lower bound: a low rate may reflect solver weakness, not missing information. To separate the two, also run the exact identifiability check (section 8). It asks whether the feedback from a game's guesses leaves only the true target tuple, which does not depend on how well the agent uses that feedback.
4. **Data:** finalize the guess dictionary and target list, then set the frequency threshold, splits, and weights, informed by step 3.
5. **Baselines:**
   - Random dictionary guesser (the floor).
   - Consistent guesser: guess a target-list word consistent with all feedback so far under some cipher consistent with that feedback.
   - Exact solver (`ExactSolver`): the particle solver with particles that are *exactly* consistent. A CP-SAT model of the whole game (cipher, one word per target, all feedback) generates them. Coverage cuts force each new solution outside every current particle's consistent-word product. If the model becomes infeasible, the particles cover the whole posterior support.
   - Particle solver over (cipher, targets), from step 3. Each particle is a decoder (the inverse cipher) plus one working word per target. Weights use the decoder alone, with targets summed out exactly: for a given decoder, each target's consistent words can be counted directly. Feedback is deterministic, so 0/1 reweighting alone collapses, and plain cipher-swap MCMC stalls after about 7 guesses. Rejuvenation therefore moves through word space: replace one target's working word, refit the decoder to the new words by transposition Metropolis (cheap, since only N words are scored), and accept on the total violation. Choose guesses by expected information gain over hypotheses drawn from the particles, plus a bonus for likely solves.
6. **Learned model:**
   - Score each dictionary word from its letters with a shared character-level encoder, not with per-index weights. Otherwise the policy learns that test-pool words never win.
   - Warm-start with behavior cloning or DAgger from the particle solver, since a random policy almost never earns a solve reward.
   - Curriculum: identity cipher, small N, smaller dictionary, and larger budget first.
   - Fine-tune with PPO. Mask already-guessed words.
   - Try an LM-based policy later. This is where rejections and R_max matter.
7. **Evaluate and ablate:** sections 8 and 9.
8. **Write-up.**

The core difficulty is cipher inference. Track it separately from target identification where possible (section 8).

## 8. Evaluation

- Held-out test-pool targets, fixed seeds shared across methods, paired comparisons. Choose the number of games per N so CIs stay useful when solve rates are near 0.
- Report by N and by length group. Metrics: full solve rate, per-target solve rate, guesses used, and rejection rate (text policies only). Include confidence intervals.
- **Identifiability:** after the first B − N guesses, and at the end, is any word tuple other than the true one consistent with all feedback under some cipher? It is checked exactly with CP-SAT and recorded as unique, ambiguous, or unknown on timeout. "Unique after B − N" means an agent with perfect inference could have won with that game's guesses.
- **Cipher-inference metric:** per step, the fraction of target letters whose cipher mapping is correct under the method's best estimate. Letters absent from the targets never affect feedback and are excluded. Read directly from the particle solver. For the learned model, use a trained probe.
- Compare random, consistent guesser, particle solver, and learned model, each with the random cipher and the identity cipher.

## 9. Ablations and Open Items

| Ablation | Purpose |
|---|---|
| Identity cipher (train and test) | Cost of the cipher, for both the particle solver and the learned model |
| Match total only (merge green and yellow) | Value of positional information |
| Per-position colors | Upper reference for feedback richness |
| Fixed cipher across games | Memorizing vs. inferring |
| Varying N and budget | Difficulty scaling |
| Hidden target lengths (future) | Need for length information |

**Resolved:**

- N uniform over {4..8}, one cipher per game.
- R_max = 20.
- Budget B(N) = N + 16: the exact solver wins 84% of games (50 games per N, test pool), with room for a learned agent above and learning signal below.
- Data: Zipf ≥ 3.0 targets, 80/10/10 pools per length (seed 0, frozen), equal length weights (section 5).

**Open:** none blocking. Choices for the learned model (architecture, training budget) are part of plan step 6.

## 10. Code

Status: plan steps 1–5 are done: spec, data (final and frozen), environment, solvability and budget study, and algorithmic baselines. The game's budget is B = N + 16. The tests pass (84). Next is the learned model (step 6). Results and next steps are below.

```
data/
  raw_data.txt            source word list (ENABLE, public domain)
  words.tsv               master file: word, length, zipf (120,572 words, length 5-10)
  by_length/len05..10.txt the same words split by length, one per line
  stats.json              counts per length, and target counts at Zipf 2.5/3.0/3.5/4.0
  splits/{train,val,test}.txt  frozen target pools (final)
  manifest.json           final data settings, counts per length, SHA-256 checksums
scripts/
  build_dictionary.py     rebuilds words.tsv and by_length/ (needs `wordfreq`)
  freeze_data.py          writes/verifies data/splits and manifest.json (--check)
  evaluate.py             paired evaluation of the baselines (section 8 metrics)
  identifiability.py      post-hoc identifiability checks on saved results, at any guess count
  run_budget.sh           budget run on half the machine (resumable), e.g. `scripts/run_budget.sh 16 50 exact`
mystic_wordle/
  scoring.py    score(guess, ciphered_target) -> Feedback(green, yellow)
  cipher.py     Cipher, CipherSampler (modes: random / identity / fixed)
  dictionary.py Dictionary (guesses, targets, target_pool, split), TargetSampler
  game.py       GameState, WordleEnv, Rewards, default_budget
  baselines/
    agents.py   Agent interface; RandomGuesser, ConsistentGuesser, ParticleSolver
    search.py   CipherSearch: particles, SMC update, candidate queries; WordTable
    kernels.py  numba kernels: constraint checks, search moves, feedback entropy
    exact.py    CP-SAT game model, ExactSolver, alternative_exists (identifiability)
tests/          pytest suite (section 6, plus kernel-vs-reference and agent tests)
results/
  budget20/     B = N + 20 study: 20 games per N, exact + particle, identifiability checks
  budget16/     B = N + 16 confirmation: 50 games per N, exact solver
```

Final settings: see section 5 (data) and section 2 (rules: B = N + 16, R_max = 20).

Usage:

```python
from mystic_wordle import Dictionary, TargetSampler, WordleEnv, CipherSampler

d = Dictionary.load()                      # final target list (Zipf >= 3.0)
train = d.target_pool("train")             # frozen pool from data/splits/train.txt
env = WordleEnv(d, TargetSampler(train), CipherSampler("random"))   # budget N + 16
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step("sheets")
```

One seed drives N, the targets and the cipher through separate random streams, so the same seed gives the same targets under the identity and random cipher modes. `reset(targets=..., cipher=...)` overrides sampling for tests and analyses. `env.state` exposes the hidden state for analysis; agents must use only the observation.

```
pip install -r requirements.txt
python -m pytest                     # ~2 min; the first run adds 1-2 min to compile the numba kernels
python scripts/freeze_data.py --check
python scripts/build_dictionary.py   # optional: rebuild words.tsv (then freeze_data.py --check must still pass)
```

### Baselines

All agents implement `reset(obs, seed)` and `act(obs) -> guess`, and read only the observation. Passing `known_cipher=Cipher.identity()` gives the identity-cipher ablation, where the cipher is part of the rules.

- `RandomGuesser`: uniform guess-dictionary words, no repeats (the floor).
- `ConsistentGuesser`: keeps one (cipher, words) state. After each guess it searches from the previous state until the state fits all feedback or the move budget runs out, then guesses a consistent word for the unsolved target with the fewest consistent words.
- `ParticleSolver`: SMC over decoders as described in section 7, step 5. Each turn it reweights, resamples when the effective sample size falls below half, and runs search moves, with extra moves for particles that are still inconsistent. It then draws 256 hypotheses and scores 1,000 random dictionary words plus every hypothesised word by feedback entropy + 3 × P(solve). It commits to a word when P(solve) ≥ 0.95 and at least 4 particles agree, and goes only for solves once the budget left equals the number of unsolved targets.

- `ExactSolver`: `ParticleSolver` with exactly consistent particles. Each turn it does four things:
  1. Re-checks old particles against the new feedback and gives broken ones a short local repair.
  2. Solves the CP-SAT game model within `cp_seconds` (default 10), adding a coverage cut after each solution so the next one is a word tuple no particle explains yet.
  3. Spawns local variants of each consistent particle with the cheap word moves.
  4. Weights particles by the product of their targets' consistent-word counts.

  If the cut model becomes infeasible, `complete` is set, the particles cover every consistent tuple, and the agent may commit to a solve without the 4-particle support rule. Every CP-SAT solution is re-checked by the numba kernel.

Search effort is set by `n_particles`, `moves`, `rescue_moves` and `move_params` (and `cp_seconds`, `max_particles` for the exact solver). The defaults keep a random-cipher game at roughly 1–5 minutes on one core.

The CP-SAT model uses one table constraint per target (word index ↔ plaintext letters), element constraints for the ciphered letters, and reified equalities for the feedback counts. Tuned parameters: one worker, probing level 0, linearization level 0, about 2× faster than the defaults. A one-hot boolean formulation and two workers were both slower. Search dominates the time, not presolve: about 5 s and 10k conflicts per solution at N = 4 with 7–13 guesses. A plain depth-first search with count bounds explored 10M nodes without finding any solution.

```
python scripts/evaluate.py --agents random consistent particle --cipher identity --n 4 5 6 7 8 --games 20 --workers 2
python scripts/evaluate.py --agents particle --cipher random --n 4 --games 50 \
    --agent-kwargs '{"particle": {"n_particles": 32, "rescue_moves": 100}}'
python scripts/evaluate.py --agents exact --cipher random --n 4 --games 4 --split val --identifiability 60
python scripts/evaluate.py --agents exact --cipher random --n 4 5 6 7 8 --games 10 --split train \
    --record-decisions --out results/demos/train-part1.jsonl        # expert data for imitation learning
python scripts/identifiability.py results/budget20/n4.jsonl --at 8 12 16 end --seconds 60 --out /tmp/id.jsonl
python scripts/identifiability.py --summarize /tmp/id.jsonl
python scripts/evaluate.py --summarize results/budget16/all.jsonl
scripts/run_budget.sh 16 50 exact            # half the machine, resumable
```

`--record-decisions` stores one record per turn: mode (`probe`, `commit` or `forced`), the guess, its entropy and P(solve), the top 5 alternatives, the number of consistent particles, whether coverage was complete, and the top 5 posterior words per unsolved target. The observation itself is not stored, because `replay(env, seed, n, guess_list)` in `scripts/evaluate.py` rebuilds it exactly.

Game seeds depend only on (N, game index), so every agent and both cipher modes play the same targets.

### Early pilots (historical: B = N + 10, earlier code)

These runs came before the exact solver's final form and the budget decision. They are superseded by the budget studies below, and their raw files were removed in the cleanup.

Identity cipher (cipher known), 20 games per N:

| agent | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|
| random | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| consistent | 20/20, 7.3 guesses | 20/20, 8.6 | 20/20, 9.1 | 20/20, 10.2 | 20/20, 10.8 |
| particle | 20/20, 7.2 guesses | 20/20, 8.3 | 20/20, 9.2 | 20/20, 10.0 | 20/20, 11.0 |

Random cipher, 4 games per N (very small; the CIs span 0–0.49):

| agent | N=4 | N=8 | final cipher accuracy | s/game |
|---|---|---|---|---|
| random | 0/4 | 0/4 | - | 0 |
| consistent | 0/4, 0 targets | 0/4, 0 targets | 0.06–0.08 | 10–33 |
| particle | 0/4, 0 targets | 0/4, 0 targets | 0.10 | 57–162 |

Exact solver vs particle solver, random cipher. 4 games at N = 4 and 3 at N = 8, identifiability checks at 60 s each. Fractions count only checks that finished within the time limit:

| agent | N | per-target solve | final cipher accuracy | unique after B − N guesses | unique at end of game | s/game |
|---|---|---|---|---|---|---|
| particle | 4 | 0/16 | 0.10 | 0/4 | 2/4 | 96 |
| exact | 4 | 0/16 | 0.17 | 0/4 | 1/4 | 138 |
| particle | 8 | 0/24 | 0.11 | 0/1 (+2 timeouts) | 2/3 | 230 |
| exact | 8 | 3/24 (all in one game) | 0.32 | 0/2 (+1 timeout) | 2/3 | 217 |

Exact-solver internals:
- It averaged 20 consistent particles per turn at N = 4 and 11–15 at N = 8.
- At N = 8 the 10 s CP-SAT budget often ran out, leaving no consistent particle on 9–13 turns per game (0–6 at N = 4).
- Coverage was complete on 2 turns, both in the game where it solved 3 targets.


### Budget study (B = N + 20, 20 games per N, random cipher, macOS, 8 cores / 8 GB)

| agent | N | full solve (95% CI) | per-target solve | guesses used | final cipher acc | s/game |
|---|---|---|---|---|---|---|
| exact | 4 | 0.95 (0.76–0.99) | 0.99 | 17.9 | 0.88 | 158 |
| exact | 5 | 0.90 (0.70–0.97) | 0.97 | 19.1 | 0.87 | 192 |
| exact | 6 | 1.00 (0.84–1.00) | 1.00 | 19.4 | 0.89 | 138 |
| exact | 7 | 0.85 (0.64–0.95) | 0.95 | 21.4 | 0.90 | 164 |
| exact | 8 | 0.90 (0.70–0.97) | 0.98 | 21.4 | 0.87 | 151 |
| particle | 4–8 | 0.00 | ≤ 0.03 | full budget | 0.05–0.08 | 108–238 |

Exact-solver per-target solve rate by target length: 0.94 (L = 5), 0.98, 0.99, 0.99, 0.96, 0.99 (L = 10). Short words are slightly harder, since they carry fewer letters to anchor the cipher.

Wins within a smaller budget B = N + e, read off the guess counts of the exact solver's won games. This approximates a rerun at that budget, because the solver only plays differently in the last few guesses.

| N | e=10 | e=12 | e=14 | e=16 | e=18 | e=20 |
|---|---|---|---|---|---|---|
| 4 | 0.15 | 0.40 | 0.65 | 0.75 | 0.90 | 0.95 |
| 5 | 0.10 | 0.55 | 0.60 | 0.70 | 0.85 | 0.90 |
| 6 | 0.20 | 0.35 | 0.65 | 0.85 | 1.00 | 1.00 |
| 7 | 0.05 | 0.35 | 0.65 | 0.75 | 0.85 | 0.85 |
| 8 | 0.20 | 0.35 | 0.70 | 0.90 | 0.90 | 0.90 |
| **pooled (100 games)** | **0.14** | **0.40** | **0.65** | **0.79** | **0.90** | **0.92** |

Identifiability: the share of the exact solver's games whose targets are unique after k guesses (CP-SAT, 120 s per check; timeouts excluded). The exact solver's own guesses include solves, which reveal the cipher.

| N | k=10 | k=12 | k=14 | k=16 | k=18 | k=20 | k=22 | k=24 | median first unique k |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1/20 | 4/20 | 8/20 | 13/20 | 17/20 | 17/20 | 18/20 | 19/20 | 16 |
| 5 | 3/20 | 3/20 | 5/20 | 11/20 | 13/20 | 15/20 | 17/20 | 18/20 | 16 |
| 6 | 1/18 | 5/19 | 6/19 | 8/19 | 12/20 | 14/20 | 18/20 | 20/20 | 18 |
| 7 | 3/15 | 5/19 | 5/20 | 6/19 | 9/20 | 12/20 | 15/20 | 16/20 | 19.5 |
| 8 | 2/15 | 5/18 | 7/19 | 9/20 | 9/20 | 12/20 | 18/20 | 18/20 | 20 |

Checks at k ≤ 10 often time out for N ≥ 6: CP-SAT cannot find an alternative within 120 s even when many exist, so early timeouts almost certainly mean ambiguous. The particle solver's own probes become unique *earlier* in many games (N = 8, k = 12: 11/20; N = 5, k = 12: 10/19), but its share then plateaus at 40–75% and it solves almost nothing.

Raw files: `results/budget20/` (n*.jsonl, ident-n*.jsonl, summary.txt). No game crashed in the N = 5 and 6 runs (no `error` rows). CP-SAT errors that were caught and retried are listed in `turn_stats[*].solver_errors`.

### Budget confirmation (B = N + 16, 50 games per N, exact solver, test pool)

Run with `scripts/run_budget.sh 16 50 exact` (macOS, 2 processes, 4 h 42 m). Raw files: `results/budget16/`.

| N | full solve (95% CI) | per-target solve | guesses used | final cipher acc | s/game |
|---|---|---|---|---|---|
| 4 | 0.84 (0.71–0.92) | 0.93 | 17.2 | 0.84 | 120 |
| 5 | 0.76 (0.63–0.86) | 0.87 | 18.7 | 0.77 | 138 |
| 6 | 0.90 (0.79–0.96) | 0.96 | 19.1 | 0.86 | 133 |
| 7 | 0.84 (0.71–0.92) | 0.94 | 20.3 | 0.85 | 142 |
| 8 | 0.84 (0.71–0.92) | 0.92 | 21.1 | 0.87 | 144 |
| **all (250)** | **0.84** | | | | |

- **Matches the estimate.** The N + 20 study predicted 70–90% for N + 16, and the direct run gives 84%. Reading wins off guess counts is a reliable shortcut.
- **No trend with N.** N = 5 is lowest, but its interval overlaps the others.
- **Most losses are near-misses.** 16 of 41 losses ended with one target unsolved.
- **The rest are mostly search failures.** 9 games solved nothing or almost nothing. In those, many turns had no consistent particle because CP-SAT hit its 10 s limit, especially at N = 7–8: 11–13 turns per lost game, against about 6 in won games. Lost games also reached complete coverage later (median guess 18, against 13 for wins).
- **Length:** per-target solve rate rises from 86% (5 letters) to 97% (10 letters).
- **CP-SAT crash:** 11 crashes in 8 games, all caught by the retry; 7 of those 8 games were won.

### Findings so far

1. **Without the cipher the game is easy.** Both search baselines win every game in about N + 3 guesses.
2. **With the cipher, the exact solver plays well and the particle solver does not.** The exact solver wins 84% at N + 16 and 85–100% at N + 20, with about 85% of the targets' cipher letters right. The particle solver wins none, because approximate particles never keep up with the posterior.
3. **Difficulty is roughly flat in N once the budget is written as N + e.** Exact solver at N + 16 (direct run): 84%. Pooled estimates: about 12% at N + 10, 36% at N + 12, 65% at N + 14.
4. **Inference, not just information, still limits play.** The particle solver's guesses often pin the targets down by guess 12, yet it solves almost nothing. The exact solver's losses come mostly from CP-SAT timeouts at larger N.

### Decision: B(N) = N + 16

The exact solver wins 84%. That leaves room above for a learned agent to beat it, and room below for a weaker agent to still get learning signal. This resolves the target-solve-rate item in section 9.

### Next steps

1. **Learned model (plan step 6).**
   - Encode the observation, and score the 120k-word dictionary through letter features.
   - Generate exact-solver demonstrations on the **train** pool with `--record-decisions`. That costs about 2.3 core-minutes per game, so 1,000 games is about 40 core-hours, or about 20 hours on half of an 8-core Mac. Store them under `results/demos/`, which git ignores.
   - Train by behaviour cloning, then DAgger and PPO, with a curriculum: identity cipher → partial cipher → full cipher.
   - Make tuning decisions on the **val** pool.
2. **Optional: stronger reference solver.** Give CP-SAT more time on turns where no particle is consistent. This targets the 9 search-failure losses. It changes the reference, so rerun the 50-game confirmation if it is adopted.
3. **Final evaluation (after step 6):** all methods on the same seeds on the **test** pool, more games per N, plus an identity-cipher reference at N + 16. Report wall-clock cost per game next to win rates.
4. **CP-SAT crash (macOS / Python 3.14, ortools 9.15.6755):** `GameModel.solve` retries without hints and otherwise treats the turn as a timeout. Recurrences are logged in `turn_stats[*].solver_errors`.

### Repository hygiene

`.gitignore` keeps the virtual environment, caches, logs, zips and generated training artefacts (`results/demos/`, checkpoints, run directories) out of git. Study results (`results/budget*/`) are small and tracked.
