# Deviations from PROTOCOL.md

Every change to the protocol after the `protocol-v1` tag goes here, one entry each, with:
date, what changed, why, and whether any tuning / re-evaluation / final results had been seen.

Also record anything unexpected seen in the pilot and any restart, crash or sleep gap that matters
(the tuning script also logs these automatically in `experiment/logs/events.jsonl`).

## Entries

### D1 - tie-break for the re-evaluation selection (PROTOCOL.md Section 5.3)

- Date: <fill in the date you commit this>
- What: Section 5.3 selects, among the 5 re-evaluated configurations, the one with the highest mean final return,
  but gives no rule for exact ties. New rule: exact ties are broken by the LOWER tuning trial number (the same
  rule Section 5.3 already uses to choose the top 5). Variance plays no role.
- Why: many configurations reach exactly 500.0 on every seed (the CartPole ceiling), so exact ties are likely.
- Results seen when this was written: tuning results, yes (all 576 runs). Re-evaluation results: no.
  Final-evaluation results: no.
- Effect on the design: none besides making the selection deterministic. The top-5 sets themselves follow from the
  existing tie-break rule: Adam trials 0, 1, 3, 5, 6; Lion trials 5, 9, 14, 28, 35; LPECO trials 0, 5, 12, 22, 24.

### D2 - two additional pre-defined SECONDARY metrics (Section 6)

- Date: <fill in the date you commit this>
- What: for every final-evaluation run (20 greedy checkpoints), also compute (a) the late-window mean: the mean of the
  10 checkpoint means from step 275,000 to 500,000 (second half of training), and (b) the late-window minimum: the
  smallest of those 10 checkpoint means. Report their SD, IQR and the LPECO / Lion SD ratio with the same BCa interval
  as SECONDARY results. They are NOT part of the decision rule in Section 2, which is unchanged.
- Why: the tuning log and the pilot show that final return often sits at the ceiling (500) for many configurations, so
  the primary metric may be non-discriminating (Section 2). These metrics register within-run dips after the ceiling
  is first reached, and they are fixed now, before any final run, so they count as pre-specified rather than
  post hoc. The "second half of training" window was chosen as a simple split that does not depend on pilot curves.
- Results seen when this was written: tuning results, yes; pilot (descriptive), yes; final evaluation, no.
- Effect on the design: none on the primary outcome or the decision rule.

### D3 - operating characteristics of the decision rule (INFORMATIONAL: no change to the protocol)

- Date: <fill in the date you commit this>
- What: a Monte Carlo study (normal data, 30 seeds per group, B = 800, 300 replications; `analysis_lib.py` as committed)
  measured how the rule of Section 2 behaves. Results:
  - No true difference (both SD 60): condition 1 alone rejected in 5.3% of replications, the permutation test alone
    in 4.0%, the full rule in 2.0% (conservative).
  - Condition 3 (non-inferiority, Delta = 25) with EQUAL true means: passed in 100% of replications when both SD = 15,
    90% at SD 30, 41% at SD 60, 28% at SD 90. With a large spread, the one-sided 95% lower bound of the IQM
    difference is 20-25 points below its estimate, so condition 3 often fails even when LPECO is not worse.
  - Full-rule power with a true SD of 25 (LPECO) vs 70 (Lion): about 0.59 if the means are equal and about 0.28 if
    LPECO is 10 points lower. (The earlier power figures in Section 2 ignored condition 3.)
- Why it is logged: so that a "no evidence for H1" result can be read correctly. When it comes from condition 3 failing
  at a large spread, the sensitivity analysis at Delta = 10 and 50 (already reported) is the relevant evidence.
- Results seen when this was written: tuning results, yes; final evaluation, no.
- Effect on the design: none.

### D4 - process deviation: re-evaluation and final stages ran before their code and selection were committed

- Date: 2026-11-12
- What happened: `reeval.py` (selection written to `experiment/selected_configs.json` at 18:56:30) and then `final.py`
  (first result file 19:02:48, last 20:34:16) were run while `reeval.py`, `final.py`, `stages_lib.py`, `analysis_lib.py`,
  `analyze.py`, `pilot.py`, `selected_configs.json` and the produced results were still uncommitted. `final.py` was started with
  `--confirm-selection-committed`, which was not yet true. The protocol asks for the selection and the tested analysis code to be
  committed before the final runs (Sections 5 and 7).
- What was already true: D1 and D2 had been committed (2026-11-11 10:07) and the tuning results had been committed unmodified
  (2026-11-11 09:58). The selection follows deterministically from them by the rule of Section 5.3 and D1. The unit tests
  (`test_stages_lib`, `test_analysis_lib`) and `analyze.py --synthetic` had passed on the experiment machine before the final
  runs started. `final.py` prints no returns.
- Results seen when this was written: tuning and re-evaluation results, yes (printed by `reeval.py`). Final-evaluation results: no
  (the only thing run on them is `analyze.py --check-only`, which prints structure and no returns).
- Remedy: all code and all results are committed unmodified (tags `final-runs-done` and `analysis-frozen`) BEFORE `analyze.py` is
  run on the final runs, and no line of `analysis_lib.py` or `analyze.py` is changed after that.
- Effect on the design: none. It only reduces how well the order of events can be verified from timestamps.
