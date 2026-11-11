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
