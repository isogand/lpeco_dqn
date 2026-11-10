# Protocol v1 — Does LPECO's correction term reduce across-seed variance in DQN training?

**Status:** v1, ready to commit. The items previously marked CONFIRM are resolved as proposed; the author must
approve them before the tag. Date: 2026-11-08.
**Rule:** this file is committed (and tagged `protocol-v1`) BEFORE any tuning, re-evaluation or final run.
After that, any change goes into `DEVIATIONS.md` with the date, the reason, and whether any results had been seen.

## 1. Question and hypotheses

**Question.** Does adding LPECO's slow-moving error-correction term to Lion reduce the spread of final
performance across independent seeds when training a DQN on CartPole, without making average performance
meaningfully worse?

- **Main comparison:** LPECO vs Lion. Lion is the same `LPECO` class with `correction_strength = 0`
  (verified identical in tests), so any difference comes from the correction term (plus its extra tuned
  hyperparameters, see Limitations).
- **Reference:** Adam. Reported descriptively; no hypothesis is tested against it.
- **No mechanism is claimed.** Only the empirical effect of the term is measured.

**Definitions.** *Final return* of a run = mean return of 20 greedy (epsilon = 0) evaluation episodes at the end of
training (step 500,000), on a separate environment instance with fixed evaluation seeds.

- **H0:** the standard deviation (SD) of final return across seeds is the same for LPECO and Lion.
- **H1:** the SD is smaller for LPECO.
- **Non-inferiority:** LPECO's average performance is not worse than Lion's by more than Δ = 25 return points
  (5% of the maximum return, 500). Δ is a **convention, not derived from data**. Sensitivity analysis: Δ = 10 and Δ = 50.

## 2. Decision rule (fixed in advance)

All intervals are one-sided 95%, BCa bootstrap, B = 10,000, with the two groups resampled
**independently** (seeds are not paired).

H1 is reported as **supported** only if ALL of the following hold:
1. Upper bound of the CI for the SD ratio (LPECO / Lion) is **< 1**.
2. A one-sided permutation test on the SD ratio (10,000 label permutations) gives p < 0.05.
3. Lower bound of the CI for IQM(LPECO) − IQM(Lion) is **> −Δ** (Δ = 25).

Otherwise the outcome is reported as **"no evidence for H1"**. This is *absence of evidence*, not equivalence:
with 30 seeds per group, power is limited. A Monte Carlo approximation (normal data, 30 seeds per group,
percentile bootstrap AND permutation, non-inferiority condition ignored) gave a false-positive rate of about 0.06
under no difference and power of about 0.3 for a 1.25x SD reduction, about 0.65 for 1.5x and about 0.97 for 2x.
CartPole returns are bounded and skewed, so these are rough guides only.

**Saturation rule.** If at least 90% of seeds in BOTH groups have final return >= 490, the primary metric cannot
discriminate and the result is reported as **"non-discriminating"** (neither support nor rejection).
**If LPECO is worse,** that is reported as is.

IQM = interquartile mean (`scipy.stats.trim_mean(x, 0.25)`).

**Zero or tiny SD.** If SD(Lion) = 0 in the observed data (e.g. every seed returns exactly 500), the SD ratio is
undefined: it is reported as "undefined" and H1 is NOT supported. In bootstrap and permutation resamples the
denominator SD is replaced by max(SD, 1.0) (one return point); resamples are never discarded, and the number of
resamples where this floor was applied is reported. If SD(LPECO) = 0 and SD(Lion) > 0 the ratio is 0 and is handled
normally. The raw (unfloored) ratio is always reported next to the floored one.

## 3. Algorithm and environment

- Environment: Gymnasium `CartPole-v1` (max return 500). Training and evaluation use separate environment instances.
- DQN written in PyTorch (derived from `time_dqn.py`, to be renamed `dqn.py`). Fixed hyperparameters:

| Item | Value |
|---|---|
| total steps | 500,000 (fixed; not adapted to any result) |
| network | MLP 4-120-84-2, ReLU |
| loss | MSE on TD target, hard target update |
| replay buffer | 10,000 transitions, uniform sampling, circular |
| batch size | 128 |
| discount gamma | 0.99 |
| learning starts | 10,000 steps |
| train every | 10 env steps (49,000 updates per run) |
| epsilon | 1.0 -> 0.05 linearly over the first 50% of steps |
| terminal handling | truncation is NOT treated as terminal |
| gradient clipping / weight decay | none |
| Adam | PyTorch defaults (beta1 0.9, beta2 0.999, eps 1e-8) |
| Lion / LPECO beta1, beta2 | 0.9, 0.99 (fixed, equal for both, passed explicitly) |

These defaults were written from memory of the CleanRL CartPole DQN settings, not copied from a file, so the
implementation is described as "DQN with CleanRL-inspired defaults", not "CleanRL's DQN".

## 4. Optimizers and search space (frozen before any run)

| Hyperparameter | Adam | Lion | LPECO |
|---|---|---|---|
| learning rate | log-uniform [5e-5, 2e-3] | log-uniform [1e-5, 1e-3] | log-uniform [1e-5, 1e-3] |
| target network frequency (env steps) | {250, 500, 1000, 2000} | same | same |
| correction_strength | – | fixed 0 | uniform [0.01, 0.4] |
| error_decay | – | – | uniform [0.98, 0.999] |

- Lion and LPECO share the same learning-rate range (Lion is LPECO with `cs = 0`). Adam has its own because
  sign-based updates move every weight by exactly `lr` per step.
- Deviations from the LPECO paper's search ranges (its Appendix A: `correction_strength` in [0.01, 0.5],
  `error_decay` in [0.98, 1.0]), on purpose: `correction_strength` upper bound 0.4 (paper: 0.5; for a
  constant gradient the update scales like (1 - 2 * cs), so it vanishes at 0.5 and reverses above it);
  `error_decay` upper bound 0.999 (paper: 1.0; gamma = 1 freezes the error EMA at zero, so LPECO would silently
  become Lion).
- **Edge rule.** If the selected learning rate falls in the outer 10% (log scale) of its range, it is reported
  under Limitations. Edges: Lion/LPECO lr < 1.58e-5 or > 6.3e-4; Adam lr < 7.2e-5 or > 1.38e-3.
- The search space is not changed after seeing any result.

## 5. Stages, seeds and run counts

Seed ranges are disjoint and asserted in code.

| Stage | Seeds | Runs |
|---|---|---|
| Pilot (Adam, default config, descriptive only) | 9000–9004 | 5 |
| Tuning | 0–999 (trial k uses seeds 3k, 3k+1, 3k+2, identical for all optimizers) | 3 optimizers x trials x 3 seeds |
| Re-evaluation of top 5 configs | 1000–1999 (config j uses seeds 1000+5j … 1000+5j+4) | 3 x 5 x 5 = 75 |
| Final evaluation | 2000–2999 (seeds 2000–2029) | 3 x 30 = 90 |
| Exploratory: best LPECO config with `cs = 0` | 2000–2029 | 30 |

Notes. Each tuning trial uses its own seeds (instead of one shared set) so the search cannot overfit a few seeds,
at the price of a noisier objective. In the final stage the same 30 seeds are used for all optimizers, so weight
initialisations are shared across groups; the analysis treats the groups as independent, which is conservative.

**5.1 Pilot.** Adam, lr 2.5e-4, target frequency 500, 500k steps, greedy evaluation every 25k steps. Purely
descriptive (pipeline check and learning curves). The horizon stays 500k regardless of what it shows. Pilot
results are never used in the analysis.

**5.2 Tuning.** Optuna, TPE sampler (sampler seed 0, `n_startup_trials = 10`), one study and one SQLite
file per optimizer. Objective: **mean final return over the 3 tuning seeds**, identical for all three optimizers
(variance is NOT part of the objective). Only a final greedy evaluation is needed per tuning run.
Two phases, decided now:
- **Phase 1:** 48 trials per optimizer.
- **Phase 2:** trials 49–64 per optimizer, run only if the throughput rule holds: at 2 hours after tuning starts,
  at least 130 tuning runs are complete (about 20% below the measured rate of ~82 runs/hour). Elapsed time is
  wall-clock time minus any gap longer than 5 minutes in which no run completed (sleep, crash, restart); such gaps
  are logged. The check is made once and its value is logged. If it fails, tuning stops at 48 trials for ALL three optimizers.
Budgets are always equal across optimizers. Resuming after an interruption re-creates the sampler with the same
seed, so a resumed study is not bit-for-bit identical to an uninterrupted one; restarts, sleep/crash gaps and the
phase decision are logged in `logs/events.jsonl`. If an interruption happens during the first 10 (random start-up)
trials of a study, the re-seeded sampler can repeat earlier draws; such interruptions are noted in `DEVIATIONS.md`. TPE arguments other than the seed and `n_startup_trials` are the
Optuna 5.0.0 defaults.

**5.3 Re-evaluation.** For each optimizer, the 5 best trials by tuning objective (ties: lower trial number) are
re-run with 5 fresh seeds each. The configuration with the highest mean final return is selected
(not the lowest variance).

**5.4 Final evaluation.** Selected configuration per optimizer x 30 seeds, greedy evaluation every 25,000 steps
(20 checkpoints x 20 episodes). Evaluation uses a separate environment and no training RNG.

**5.5 Run counts.** 64 trials: 576 + 75 + 90 + 30 = **771**. 48 trials: 432 + 75 + 90 + 30 = **627**.

## 6. Metrics

- **Primary:** SD of final return across the 30 seeds; ratio LPECO / Lion with CI (Section 2).
- **Co-primary condition:** IQM difference with CI (non-inferiority).
- **Secondary (pre-defined):**
  - IQR of final return; fraction of seeds with final return < 250 (**weak seed**).
  - **Collapse count** per seed: number of evaluation checkpoints, after the first checkpoint with mean >= 490,
    whose mean is < 250. Report the mean per seed and the fraction of seeds with at least one collapse.
  - Brown–Forsythe test (two-sided, supplementary only).
- **Exploratory (labelled as such in any report):** the `cs = 0` re-run of the best LPECO configuration (its lr was
  tuned for LPECO, not for Lion); learning curves; any additional diagnostic logged later (e.g. dormant-neuron
  fraction, weight norms, mid-training environment change) only if added through `DEVIATIONS.md` before the
  final runs, and never used for the decision rule.
- Adam: the same statistics versus Lion and LPECO, descriptively.

## 7. Implementation and reproducibility

- Hardware/software: MacBook Pro, Intel Core i7-8550U (4 physical / 8 logical cores), Python 3.12.3, PyTorch 2.2.2,
  Gymnasium 1.0.0, NumPy 1.26.4, SciPy 1.15.2 (from `pip list` on the experiment machine); Optuna 5.0.0 (with SQLAlchemy 2.1.8; installed 2026-11-08). No package is upgraded during the experiments.
- **4 parallel worker processes**, `torch.set_num_threads(1)`. Measured total throughput after learning starts:
  ~13,650 steps/s in a short burst (4 workers; 3 workers 10,983; 5 workers 12,132; 6 workers 12,355), and a
  mean of ~11,356 steps/s in a 12-minute sustained test. Estimated duration: ~9.4 h for 771 runs, ~7.7 h for 627.
- LPECO optimizer cost measured separately: +1.4 s per 500k-step run relative to Adam (about 0.8%).
- Each run writes its own result file; completed runs are skipped on restart; the machine runs on mains power
  under `caffeinate -i`.
- Determinism checks already passed: same seed gives identical final weights (hash) alone and under 1, 3, 4, 5
  and 6 parallel processes; periodic greedy evaluation leaves the training trajectory unchanged;
  `--opt lion` and `--opt lpeco --cs 0` give identical hashes.
- The PyTorch optimizer (`lpeco_torch.py`) is verified by 19 unit tests (numpy reference, the original Keras
  class, `cs = 0` equals Lion, hand-computed values, steady-state behaviour, state-dict round trip).
- Any change to training logic must keep the weight hash for a fixed seed identical (regression test).
- Tuning code: `dqn.py` (importable refactor of `time_dqn.py`, regression-tested against the hashes measured with
  `time_dqn.py`), `tune_lib.py` with `test_tune_lib.py` (pure logic: seeds, search spaces, edge rule, active-time
  rule, phase decision), and `tune.py` (3 Optuna studies sharing one pool of 4 worker processes, one JSON file per
  run, resumable). The real tuning only starts with `--confirm-protocol-committed`.
- The analysis script is committed and tested on synthetic data BEFORE the final runs. It uses a fixed
  bootstrap/permutation seed.

## 8. Limitations (to be stated in any report)

- LPECO has two more tuned hyperparameters than Lion/Adam (and TPE searches a 4-D space with the same trial
  budget), so the comparison is budget-equal, not dimension-equal.
- Configurations are chosen to maximise **mean** return, so variance results are conditional on mean-optimal
  settings. Tuning uses only 3 seeds per trial (noisy); re-evaluation mitigates but does not remove this.
- Only CartPole, one network, one DQN variant, 500k steps. No claim about RL in general or about the mechanism.
- DQN defaults are tuned-by-convention for Adam; target frequency and lr are searched, other settings are not.
- With 30 seeds per group, power to detect moderate SD differences is low.
- Final return uses 20 evaluation episodes per seed (noisy), CartPole may saturate (non-discriminating rule).
- Edge-of-range selections (Section 4) and any deviation are reported.
- Pilot is Adam-only and says nothing about Lion/LPECO behaviour.

## 9. Out of scope

Instability monitoring/detection rules, other environments, claims of novelty (comparisons of optimizers in deep
RL already exist), claims that LPECO "works" on RL in general.

## 10. Checklist

Before the commit (all done except the author's approval):
- [x] Proposals resolved: `n_startup_trials = 10`, bootstrap B = 10,000, sampler seed 0, "CleanRL-inspired" wording.
- [x] Versions recorded (Section 7): Python 3.12.3, PyTorch 2.2.2, Gymnasium 1.0.0, NumPy 1.26.4, SciPy 1.15.2, Optuna 5.0.0.
- [x] Tuning code written and rehearsed (`tune.py --smoke`, including an interruption and a resume); `dqn.py`
  regression-tested against `time_dqn.py`; `DEVIATIONS.md` created.
- [ ] Author approves; commit this file with the code and tag `protocol-v1`.

Before the stages they belong to (not needed for the commit):
- [ ] Pilot script, then the pilot (5 Adam seeds).
- [ ] Re-evaluation and final-evaluation scripts.
- [ ] `analyze.py`, tested on synthetic data, committed before the final runs start.

## 11. Change log

- v1 (2026-11-08): initial draft.
- v1 draft rev 2 (2026-11-08): zero-SD convention, power estimates, sleep-gap handling for the 2-hour rule, seed notes, paper-range citation.
- v1 draft rev 3 (2026-11-08): Optuna 5.0.0 recorded, resume behaviour, tuning code listed.
- v1 ready to commit (rev 5, 2026-11-08): proposals resolved, checklist restructured, resume note added.
- v1 draft rev 4 (2026-11-08): Python 3.12.3 recorded. On the experiment machine `test_tune_lib` (22 tests) and
  `test_dqn` (6 tests: hashes of `time_dqn.py` reproduced, evaluation does not change training) pass, and a
  `tune.py --smoke` rehearsal with an interruption and a resume completed.
