# Analysis report (pre-registered rule, PROTOCOL.md Section 2)

Bootstrap: BCa, B = 10000, seed 12345, groups resampled independently. Non-inferiority margin Delta = 25.0 (convention).

## Primary: LPECO vs Lion (30 seeds each)

- DECISION: **no evidence for H1**
- SD of final return: LPECO 125.06, Lion 161.59; ratio (floored) 0.774, raw 0.774
- Condition 1, upper 95% bound of the SD ratio < 1: False (bound 1.251)
- Condition 2, permutation p < 0.05: False (p = 0.1496)
- Condition 3, IQM(LPECO) - IQM(Lion) = 13.33, lower 95% bound -23.05 > -Delta: True
- Saturation: share of seeds >= 490: LPECO 0.70, Lion 0.67 (rule: non-discriminating if both >= 0.90): False
- LPECO worse than Lion (upper 95% bound of IQM difference < 0): False
- Sensitivity of the decision to Delta: {'10.0': 'no evidence for H1', '25.0': 'no evidence for H1', '50.0': 'no evidence for H1'}
- Brown-Forsythe (two-sided, supplementary): p = 0.4104

'no evidence for H1' is absence of evidence, not equivalence (Section 2).

## Secondary (not part of the decision rule)

- late-window mean: SD LPECO 41.46 vs Lion 54.17, ratio 0.765, upper 95% bound 1.114, permutation p 0.0662
- late-window minimum: SD LPECO 166.18 vs Lion 119.82, ratio 1.387, upper 95% bound 1.931, permutation p 0.9925

## Descriptive per group

- adam: final mean 495.4, SD 21.9, IQM 500.0, IQR 0.0, share < 250: 0.00, share >= 490: 0.93; collapses: {'never_reached_490': 0, 'mean_collapses_per_seed': 0.0, 'frac_seeds_with_collapse': 0.0}
- lion: final mean 404.3, SD 161.6, IQM 483.5, IQR 119.4, share < 250: 0.23, share >= 490: 0.67; collapses: {'never_reached_490': 0, 'mean_collapses_per_seed': 2.4, 'frac_seeds_with_collapse': 0.8333333333333334}
- lpeco: final mean 435.2, SD 125.1, IQM 496.9, IQR 28.1, share < 250: 0.10, share >= 490: 0.70; collapses: {'never_reached_490': 0, 'mean_collapses_per_seed': 0.9666666666666667, 'frac_seeds_with_collapse': 0.5}
- lpeco_cs0: final mean 268.9, SD 183.7, IQM 262.9, IQR 367.3, share < 250: 0.57, share >= 490: 0.27; collapses: {'never_reached_490': 1, 'mean_collapses_per_seed': 2.7241379310344827, 'frac_seeds_with_collapse': 0.9655172413793104}

## Adam (descriptive reference only)

- adam_vs_lion_final_sd_ratio: ratio 0.136, upper 95% bound 0.324
- adam_vs_lpeco_final_sd_ratio: ratio 0.175, upper 95% bound 0.456

## Exploratory: LPECO vs the same configuration with cs = 0

- SD LPECO 125.06 vs cs=0 183.66, ratio 0.681 (the learning rate was tuned for LPECO, not for Lion: exploratory only)
