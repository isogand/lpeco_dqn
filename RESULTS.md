# Results: does LPECO's correction term reduce across-seed variance in DQN on CartPole?

All numbers below are copied from `experiment/analysis/report.md` and `results.json`, produced by `analyze.py` with the rule
pre-registered in `PROTOCOL.md` (tag `protocol-v1`). This file was written after the analysis; it does not change it.

## Short answer

**The pre-registered primary hypothesis was not supported.** The standard deviation (SD) of final return across 30 seeds was
0.77 times smaller for LPECO than for Lion (125.1 vs 161.6), but the one-sided 95% upper bound of that ratio was 1.25 and the
permutation p-value was 0.15, so the rule reports **"no evidence for H1"**. This is absence of evidence, not evidence of no
effect: the study had low power for an effect of this size (see Limitations).

## Design in one paragraph

DQN on `CartPole-v1` (500,000 steps), three optimizers (Adam as reference, Lion, LPECO; Lion is the LPECO class with
`correction_strength = 0`), 64 TPE trials per optimizer with 3 tuning seeds each, re-evaluation of the 5 best trials per
optimizer on 5 fresh seeds, then 30 fresh final seeds (2000-2029) per optimizer plus an exploratory group (the selected LPECO
configuration with `cs = 0`). Final return = mean of 20 greedy episodes at the end of training. Decision rule (all must hold):
upper 95% BCa bound of the SD ratio LPECO/Lion below 1, permutation p below 0.05, and lower 95% bound of IQM(LPECO) - IQM(Lion)
above -25 (a convention, 5% of the maximum return).

Selected configurations (re-evaluation mean over 5 seeds in brackets):

| Optimizer | Tuning trial | Learning rate | Target update every | Other | Re-eval mean |
|---|---|---|---|---|---|
| Adam | 6 | 1.33e-4 | 250 | | 500.0 |
| Lion | 14 | 1.41e-4 | 500 | | 405.4 |
| LPECO | 22 | 3.78e-5 | 250 | cs 0.336, error_decay 0.9873 | 500.0 |

## Primary result (LPECO vs Lion, 30 seeds each)

| Quantity | Value |
|---|---|
| SD of final return | LPECO 125.06, Lion 161.59 |
| SD ratio (LPECO / Lion) | 0.774 |
| Condition 1: upper 95% BCa bound of the ratio < 1 | **False** (bound 1.251) |
| Condition 2: permutation p < 0.05 | **False** (p = 0.1496) |
| Condition 3: lower 95% bound of IQM difference > -25 | True (IQM difference +13.33, lower bound -23.05) |
| Seeds with final return >= 490 | LPECO 70%, Lion 67% (not saturated) |
| Brown-Forsythe, two-sided (supplementary) | p = 0.41 |
| Decision | **no evidence for H1**, for Delta = 10, 25 and 50 |

## Secondary and descriptive results (not part of the decision rule)

| Group | Mean | SD | IQM | IQR | Seeds < 250 | Seeds >= 490 | Collapses per seed |
|---|---|---|---|---|---|---|---|
| Adam | 495.4 | 21.9 | 500.0 | 0.0 | 0% | 93% | 0.0 |
| Lion | 404.3 | 161.6 | 483.5 | 119.4 | 23% | 67% | 2.4 |
| LPECO | 435.2 | 125.1 | 496.9 | 28.1 | 10% | 70% | 0.97 |
| LPECO with cs = 0 (exploratory) | 268.9 | 183.7 | 262.9 | 367.3 | 57% | 27% | 2.7 |

- Adam is much more stable than both sign-based optimizers: SD ratio Adam/Lion 0.136 (upper 95% bound 0.324), Adam/LPECO 0.175
  (upper bound 0.456). Descriptive only.
- Second half of training (10 checkpoints): SD of the mean return, LPECO 41.5 vs Lion 54.2 (ratio 0.765, upper bound 1.114,
  permutation p 0.066); SD of the minimum return, LPECO 166.2 vs Lion 119.8 (ratio 1.39, upper bound 1.93, p 0.99).
  The two secondary views do not agree.
- Exploratory ablation: the selected LPECO configuration run with `cs = 0` did much worse than with the correction term
  (mean 268.9 vs 435.2; SD ratio LPECO / cs=0 = 0.68). The learning rate was tuned for LPECO, not for Lion, so this may
  simply be a poorly tuned Lion; it does not identify a mechanism.

## What this does and does not show

- It does show: in this setup, with equal tuning budgets and a fixed rule, there is no evidence that LPECO reduces the
  across-seed spread of final return compared with Lion; and Adam is clearly more stable than both.
- It does not show that LPECO has no effect on variance, that it is better than Lion on average (that was not a
  pre-registered hypothesis; the IQM difference interval includes 0), or anything about reinforcement learning in general.
  The observed IQR (28 vs 119) and the share of failed seeds (10% vs 23%) are exploratory observations, not findings.

## Limitations

- Power. A Monte Carlo study (see `DEVIATIONS.md`, D3) gave about 0.3 power for a 1.25x SD reduction with 30 seeds per
  group, and the non-inferiority condition lowers it further when the spread is large. The observed reduction (1.29x) is in
  that range. Detecting 1.3x with 80% power would need roughly 90 seeds per group (normal approximation).
- LPECO has two more tuned hyperparameters than Lion and Adam under the same trial budget; configurations were chosen for mean
  return, so variance results are conditional on mean-optimal settings; tuning used only 3 seeds per trial.
- Winner's curse: the selected LPECO configuration scored 500.0 on its 5 re-evaluation seeds and 435.2 on the 30 final seeds
  (Adam 500.0 -> 495.4; Lion 405.4 -> 404.3). Fresh final seeds are what makes the comparison fair.
- One environment, one network, one DQN variant, 500k steps; one machine and software stack.
- Many tuning trials hit the ceiling (500) on all 3 tuning seeds (Adam 32 of 64, Lion 12, LPECO 13), so the tuning objective
  could not separate good from very good configurations; ties were broken by the lower trial number (D1).

## Process record

Protocol committed before any tuning (`protocol-v1`); tuning results committed unmodified (`tuning-done`); deviations D1-D4
in `DEVIATIONS.md`, notably D4: the re-evaluation and final stages ran before their code and `selected_configs.json` were
committed (the analysis code was frozen in tag `analysis-frozen` before `analyze.py` was run, and no final-run result was
inspected before that). Reproduce the analysis with `python3 analyze.py` (fixed bootstrap seed 12345).

## Possible follow-up (not done)

A larger, separately pre-registered run with new seeds (it would be a new experiment and could not replace the result above),
and a Lion baseline tuned at LPECO's learning rate to make the `cs = 0` ablation fair.
