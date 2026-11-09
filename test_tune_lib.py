"""Tests for tune_lib.py (no torch / optuna needed).

    python3 -m unittest -v test_tune_lib
"""
import json
import tempfile
import unittest
from pathlib import Path

import tune_lib as T


class SeedTests(unittest.TestCase):
    def test_trial_seeds(self):
        self.assertEqual(T.tuning_seeds(0), [0, 1, 2])
        self.assertEqual(T.tuning_seeds(5), [15, 16, 17])
        self.assertEqual(T.tuning_seeds(63), [189, 190, 191])

    def test_no_overlap_between_trials(self):
        seen = set()
        for k in range(64):
            s = set(T.tuning_seeds(k))
            self.assertFalse(seen & s)
            seen |= s
        self.assertLess(max(seen), 1000)

    def test_seed_range_guard(self):
        self.assertEqual(T.tuning_seeds(332), [996, 997, 998])  # last allowed trial
        with self.assertRaises(ValueError):
            T.tuning_seeds(333)  # would use seeds 999, 1000, 1001


class SpaceTests(unittest.TestCase):
    def test_bounds_match_protocol(self):
        self.assertEqual(T.SPACES["adam"]["lr"], ("float", 5e-5, 2e-3, True))
        self.assertEqual(T.SPACES["lion"]["lr"], ("float", 1e-5, 1e-3, True))
        self.assertEqual(T.SPACES["lpeco"]["lr"], T.SPACES["lion"]["lr"])
        self.assertEqual(T.SPACES["lpeco"]["cs"], ("float", 0.01, 0.4, False))
        self.assertEqual(T.SPACES["lpeco"]["error_decay"], ("float", 0.98, 0.999, False))
        for opt in T.OPTIMIZERS:
            self.assertEqual(T.SPACES[opt]["target_every"], ("cat", [250, 500, 1000, 2000]))

    def test_dimensions(self):
        self.assertEqual(len(T.SPACES["adam"]), 2)
        self.assertEqual(len(T.SPACES["lion"]), 2)
        self.assertEqual(len(T.SPACES["lpeco"]), 4)

    def test_lion_is_lpeco_with_cs_zero(self):
        self.assertEqual(T.fixed_params("lion")["cs"], 0.0)
        self.assertEqual(T.fixed_params("adam")["cs"], 0.0)
        self.assertEqual(T.fixed_params("lpeco"), {})

    def test_cs_upper_bound_is_below_steady_state_reversal(self):
        self.assertLess(T.SPACES["lpeco"]["cs"][2], 0.5)
        self.assertLess(T.SPACES["lpeco"]["error_decay"][2], 1.0)

    def test_edges(self):
        lo, hi = T.lr_edges("lion")
        self.assertAlmostEqual(lo, 1.585e-5, delta=2e-8)
        self.assertAlmostEqual(hi, 6.31e-4, delta=2e-6)
        lo, hi = T.lr_edges("adam")
        self.assertAlmostEqual(lo, 7.23e-5, delta=2e-7)
        self.assertAlmostEqual(hi, 1.383e-3, delta=2e-6)
        self.assertTrue(T.lr_at_edge("lion", 1.2e-5))
        self.assertTrue(T.lr_at_edge("lion", 8e-4))
        self.assertFalse(T.lr_at_edge("lion", 1e-4))
        self.assertTrue(T.lr_at_edge("adam", 6e-5))
        self.assertFalse(T.lr_at_edge("adam", 2.5e-4))


class ActiveTimeTests(unittest.TestCase):
    def test_no_gaps(self):
        finishes = [100, 250, 400, 550]
        self.assertAlmostEqual(T.active_elapsed(0, finishes, 600), 600)

    def test_long_gap_is_excluded(self):
        # 0..200 active, then a 1000 s gap (sleep), then 1200..1400 active, now = 1450
        finishes = [100, 200, 1200, 1300, 1400]
        self.assertAlmostEqual(T.active_elapsed(0, finishes, 1450), 200 + 200 + 50)

    def test_trailing_gap_excluded(self):
        self.assertAlmostEqual(T.active_elapsed(0, [100, 200], 5000), 200)

    def test_exact_threshold_gap_counts(self):
        self.assertAlmostEqual(T.active_elapsed(0, [300], 300), 300)  # gap == 300 counts

    def test_long_gaps_listing(self):
        gaps = T.long_gaps(0, [100, 200, 1200, 1300])
        self.assertEqual(gaps, [(200, 1200)])

    def test_unsorted_input(self):
        self.assertAlmostEqual(T.active_elapsed(0, [400, 100, 250], 450), 450)


class PhaseTests(unittest.TestCase):
    def test_decision_threshold(self):
        self.assertTrue(T.make_decision(7200, 130)["proceed_to_phase2"])
        self.assertTrue(T.make_decision(7200, 200)["proceed_to_phase2"])
        self.assertFalse(T.make_decision(7200, 129)["proceed_to_phase2"])

    def test_target_trials(self):
        yes = T.make_decision(7200, 150)
        no = T.make_decision(7200, 100)
        self.assertEqual(T.target_trials(0, None), 48)
        self.assertEqual(T.target_trials(47, None), 48)
        self.assertIsNone(T.target_trials(48, None))      # wait for the decision
        self.assertEqual(T.target_trials(48, yes), 64)
        self.assertEqual(T.target_trials(48, no), 48)
        self.assertEqual(T.target_trials(60, yes), 64)

    def test_decision_ignores_returns(self):
        d = T.make_decision(7200, 140)
        self.assertEqual(set(d), {"decided_at", "active_elapsed_s", "runs_completed", "min_runs", "proceed_to_phase2"})

    def test_run_counts_match_protocol(self):
        self.assertEqual(T.run_counts(64)["tuning"], 576)
        self.assertEqual(T.run_counts(64)["total"], 771)
        self.assertEqual(T.run_counts(48)["tuning"], 432)
        self.assertEqual(T.run_counts(48)["total"], 627)


class JobTests(unittest.TestCase):
    def test_build_job_lpeco(self):
        params = {"lr": 3e-4, "target_every": 500, "cs": 0.2, "error_decay": 0.99}
        job = T.build_job("lpeco", 4, 1, params, T.PROTOCOL, "exp")
        kw = job["train_kwargs"]
        self.assertEqual(job["seed"], 13)
        self.assertEqual(kw["seed"], 13)
        self.assertEqual(kw["cs"], 0.2)
        self.assertEqual(kw["steps"], 500_000)
        self.assertEqual(kw["eval_every"], 500_000)
        self.assertEqual(kw["eval_episodes"], 20)
        self.assertEqual((kw["beta1"], kw["beta2"]), (0.9, 0.99))
        self.assertTrue(job["out_path"].endswith("exp/tuning/lpeco/trial004_seed0013.json"))

    def test_build_job_lion_forces_cs_zero(self):
        params = {"lr": 1e-4, "target_every": 1000}
        job = T.build_job("lion", 0, 0, params, T.PROTOCOL, "exp")
        self.assertEqual(job["train_kwargs"]["cs"], 0.0)
        self.assertEqual(job["train_kwargs"]["opt"], "lion")

    def test_same_seeds_for_all_optimizers(self):
        for opt in T.OPTIMIZERS:
            params = {"lr": 1e-4, "target_every": 500, "cs": 0.1, "error_decay": 0.99}
            seeds = [T.build_job(opt, 7, j, params, T.PROTOCOL, "exp")["seed"] for j in range(3)]
            self.assertEqual(seeds, [21, 22, 23])

    def test_collect_finish_times(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tuning" / "adam"
            p.mkdir(parents=True)
            (p / "a.json").write_text(json.dumps({"finished_at": 10.0}))
            (p / "b.json").write_text(json.dumps({"finished_at": 20.0}))
            (p / "bad.json").write_text("{not json")
            self.assertEqual(sorted(T.collect_finish_times(d)), [10.0, 20.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
