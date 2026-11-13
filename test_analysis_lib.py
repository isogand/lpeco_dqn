"""Tests for analysis_lib.py on synthetic data with known truth (numpy + scipy only).

    python3 -m unittest -v test_analysis_lib
"""
import unittest

import numpy as np

import analysis_lib as A

B_TEST = 2000


def clip(v):
    return np.clip(v, 0.0, 500.0)


def make_runs(finals, rng, late_noise=0.0):
    """Synthetic runs: 20 checkpoints that rise to each run's final value (optionally noisy in the late window)."""
    runs = []
    for f in finals:
        base = np.linspace(100.0, f, A.N_CHECKPOINTS)
        base[A.N_CHECKPOINTS // 2:] += rng.normal(0, late_noise, A.N_CHECKPOINTS // 2)
        base[-1] = f
        base = clip(base)
        runs.append({"final_return": float(f), "evals": [[25_000 * (i + 1), float(v)] for i, v in enumerate(base)]})
    return runs


class MetricTests(unittest.TestCase):
    def test_iqm(self):
        self.assertAlmostEqual(A.iqm(list(range(1, 9))), 4.5)             # middle half of 1..8 = 3..6
        self.assertAlmostEqual(A.iqm([0, 0, 0, 100, 100, 100, 100, 100]), 75.0)

    def test_late_window_is_second_half(self):
        evals = [[i, float(i)] for i in range(1, 21)]
        self.assertAlmostEqual(A.late_mean(evals), np.mean(range(11, 21)))
        self.assertEqual(A.late_min(evals), 11.0)

    def test_checkpoint_count_enforced(self):
        with self.assertRaises(ValueError):
            A.late_mean([[1, 1.0]] * 19)

    def test_collapse_count(self):
        up = [100.0] * 5 + [500.0] + [100.0, 400.0, 100.0] + [500.0] * 11
        self.assertEqual(A.collapse_count([[i, v] for i, v in enumerate(up)]), 2)
        never = [[i, 300.0] for i in range(20)]
        self.assertIsNone(A.collapse_count(never))
        clean = [[i, 500.0] for i in range(20)]
        self.assertEqual(A.collapse_count(clean), 0)

    def test_sd_ratio_floor(self):
        x = np.array([10.0, 20.0, 30.0])
        l = np.array([500.0, 500.0, 500.0])
        self.assertAlmostEqual(float(A.stat_sd_ratio(x[None], l[None])[0]), np.std(x, ddof=1) / 1.0)


class BcaTests(unittest.TestCase):
    def test_bound_brackets_truth_in_a_clear_case(self):
        rng = np.random.default_rng(1)
        x, l = rng.normal(0, 1, 30) + 300, rng.normal(0, 4, 30) + 300
        ub, theta = A.bca_bound(A.stat_sd_ratio, x, l, B_TEST, np.random.default_rng(2), "upper")
        self.assertLess(theta, 0.5)
        self.assertLess(ub, 0.5 + 0.2)      # true ratio 0.25
        self.assertGreater(ub, theta)

    def test_lower_bound_below_estimate(self):
        rng = np.random.default_rng(3)
        x, l = rng.normal(400, 30, 30), rng.normal(380, 30, 30)
        lb, theta = A.bca_bound(A.stat_iqm_diff, x, l, B_TEST, np.random.default_rng(4), "lower")
        self.assertLess(lb, theta)

    def test_coverage_of_one_sided_upper_bound_is_reasonable(self):
        # true SD ratio = 1 (no difference), realistic units (SD 60 return points, so the 1-point floor is irrelevant):
        # the 95% upper bound should be >= 1 in about 95% of replications (simulation measured 0.947)
        rng = np.random.default_rng(5)
        hits, reps = 0, 200
        for r in range(reps):
            x, l = rng.normal(380, 60, 30), rng.normal(380, 60, 30)
            ub, _ = A.bca_bound(A.stat_sd_ratio, x, l, 400, np.random.default_rng(1000 + r), "upper")
            hits += ub >= 1.0
        self.assertGreater(hits / reps, 0.90)


class PermutationTests(unittest.TestCase):
    def test_small_p_for_a_big_effect_and_large_p_for_none(self):
        rng = np.random.default_rng(6)
        big_x, big_l = rng.normal(0, 5, 30), rng.normal(0, 40, 30)
        self.assertLess(A.perm_p_sd_ratio(big_x, big_l, B_TEST, np.random.default_rng(7)), 0.01)

    def test_null_rejection_rate_is_near_nominal(self):
        rng = np.random.default_rng(8)
        rej, reps = 0, 300
        for r in range(reps):
            a, b = rng.normal(380, 60, 30), rng.normal(380, 60, 30)
            rej += A.perm_p_sd_ratio(a, b, 300, np.random.default_rng(2000 + r)) < 0.05
        self.assertLess(rej / reps, 0.09)        # simulation measured 0.040


class DecisionTests(unittest.TestCase):
    def test_clear_support(self):
        rng = np.random.default_rng(10)
        lpeco = clip(rng.normal(400, 20, 30))
        lion = clip(rng.normal(400, 90, 30))
        r = A.primary_analysis(lpeco, lion, B=B_TEST, rng=np.random.default_rng(11))
        self.assertEqual(r["decision"], "H1 supported")
        self.assertFalse(r["saturated"])

    def test_no_difference_is_not_supported(self):
        rng = np.random.default_rng(12)
        r = A.primary_analysis(clip(rng.normal(380, 60, 30)), clip(rng.normal(380, 60, 30)), B=B_TEST,
                               rng=np.random.default_rng(13))
        self.assertEqual(r["decision"], "no evidence for H1")

    def test_both_saturated_is_non_discriminating(self):
        lpeco = np.array([500.0] * 29 + [300.0])
        lion = np.array([500.0] * 28 + [200.0, 100.0])
        r = A.primary_analysis(lpeco, lion, B=B_TEST, rng=np.random.default_rng(14))
        self.assertTrue(r["saturated"])
        self.assertEqual(r["decision"], "non-discriminating")

    def test_zero_sd_lion_is_undefined_and_not_supported(self):
        lion = np.full(30, 500.0)
        lpeco = np.array([500.0] * 12 + [100.0] * 18)       # many below 490 -> not saturated
        r = A.primary_analysis(lpeco, lion, B=B_TEST, rng=np.random.default_rng(15))
        self.assertTrue(r["ratio_undefined"])
        self.assertIsNone(r["sd_ratio_raw"])
        self.assertTrue(r["decision"].startswith("no evidence for H1"))
        self.assertTrue(r["lpeco_worse"])

    def test_small_sd_but_much_worse_mean_fails_non_inferiority(self):
        rng = np.random.default_rng(16)
        lpeco = clip(rng.normal(300, 8, 30))
        lion = clip(rng.normal(420, 70, 30))
        r = A.primary_analysis(lpeco, lion, B=B_TEST, rng=np.random.default_rng(17))
        self.assertTrue(r["cond1_ratio_upper_below_1"])
        self.assertFalse(r["cond3_by_delta"]["25.0"])
        self.assertEqual(r["decision"], "no evidence for H1")
        self.assertTrue(r["lpeco_worse"])

    def test_sensitivity_to_delta_is_reported(self):
        rng = np.random.default_rng(18)
        lpeco = clip(rng.normal(385, 15, 30))        # about 15 points below Lion on average, much smaller SD
        lion = clip(rng.normal(400, 90, 30))
        r = A.primary_analysis(lpeco, lion, B=B_TEST, rng=np.random.default_rng(19))
        self.assertEqual(set(r["decision_by_delta"]), {"10.0", "25.0", "50.0"})


class FullAnalysisTests(unittest.TestCase):
    def test_pipeline_on_synthetic_runs(self):
        rng = np.random.default_rng(20)
        runs = {
            "adam": make_runs(clip(rng.normal(420, 70, 30)), rng, 20),
            "lion": make_runs(clip(rng.normal(400, 90, 30)), rng, 30),
            "lpeco": make_runs(clip(rng.normal(420, 20, 30)), rng, 10),
            "lpeco_cs0": make_runs(clip(rng.normal(410, 60, 30)), rng, 20),
        }
        res = A.run_analysis(runs, B=B_TEST, seed=21)
        self.assertEqual(res["primary"]["decision"], "H1 supported")
        self.assertIn("late_mean_sd_ratio", res["secondary_lpeco_vs_lion"])
        self.assertIn("exploratory_lpeco_vs_lpeco_cs0", res)
        text = A.format_report(res)
        self.assertIn("DECISION", text)
        self.assertIn("not equivalence", text)

    def test_same_seed_same_answer(self):
        rng = np.random.default_rng(22)
        runs = {k: make_runs(clip(rng.normal(400, 60, 30)), rng) for k in ("adam", "lion", "lpeco")}
        a = A.run_analysis(runs, B=500, seed=5)
        b = A.run_analysis(runs, B=500, seed=5)
        self.assertEqual(a["primary"]["ratio_ci_upper95"], b["primary"]["ratio_ci_upper95"])
        self.assertEqual(a["primary"]["perm_p"], b["primary"]["perm_p"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
