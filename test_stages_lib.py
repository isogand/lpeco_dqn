"""Tests for stages_lib.py (no torch / optuna needed).

    python3 -m unittest -v test_stages_lib
"""
import json
import tempfile
import unittest
from pathlib import Path

import stages_lib as S
import tune_lib as T

CFG = dict(lr=1e-4, target_every=500, cs=0.2, error_decay=0.99, beta1=0.9, beta2=0.99)


def write_tuning(root, opt, values_by_trial):
    d = Path(root) / "tuning" / opt
    d.mkdir(parents=True, exist_ok=True)
    for k, vals in values_by_trial.items():
        for j, v in enumerate(vals):
            seed = 3 * k + j
            (d / f"trial{k:03d}_seed{seed:04d}.json").write_text(json.dumps(
                {"seed": seed, "final_return": v, "config": dict(CFG, lr=1e-4 * (k + 1)), "job": {"trial": k}}))


class TopKTests(unittest.TestCase):
    def test_ties_go_to_lower_trial_number(self):
        with tempfile.TemporaryDirectory() as d:
            vals = {k: [500.0, 500.0, 500.0] for k in (7, 3, 11, 5, 9, 1, 13)}
            vals.update({2: [499.0, 500.0, 500.0], 4: [100.0, 100.0, 100.0]})
            write_tuning(d, "lion", vals)
            top = S.top_k(S.load_tuning_trials(d, "lion"))
            self.assertEqual([t["trial"] for t in top], [1, 3, 5, 7, 9])      # exact 500.0 ties, lowest numbers
            self.assertEqual([t["rank"] for t in top], [0, 1, 2, 3, 4])

    def test_higher_value_beats_lower_trial_number(self):
        with tempfile.TemporaryDirectory() as d:
            vals = {0: [300.0] * 3, 1: [400.0] * 3, 2: [500.0] * 3, 3: [450.0] * 3, 4: [320.0] * 3, 5: [310.0] * 3}
            write_tuning(d, "adam", vals)
            top = S.top_k(S.load_tuning_trials(d, "adam"))
            self.assertEqual([t["trial"] for t in top], [2, 3, 1, 4, 5])

    def test_incomplete_trial_raises(self):
        with tempfile.TemporaryDirectory() as d:
            write_tuning(d, "adam", {0: [1.0, 2.0, 3.0], 1: [1.0, 2.0]})
            with self.assertRaises(ValueError):
                S.load_tuning_trials(d, "adam")

    def test_near_ties_below_1e9_count_as_ties(self):
        self.assertEqual(S.tie_key(500.0, 4), S.tie_key(500.0 - 1e-12, 4))
        self.assertLess(S.tie_key(500.0, 9), S.tie_key(499.9, 1))


class SeedTests(unittest.TestCase):
    def test_reeval_seeds(self):
        self.assertEqual(S.reeval_seeds(0), [1000, 1001, 1002, 1003, 1004])
        self.assertEqual(S.reeval_seeds(4), [1020, 1021, 1022, 1023, 1024])

    def test_ranges_disjoint(self):
        tuning = set(range(0, 1000))
        reeval = {s for r in range(5) for s in S.reeval_seeds(r)}
        final = set(S.FINAL_SEEDS)
        self.assertFalse(tuning & reeval)
        self.assertFalse(tuning & final)
        self.assertFalse(reeval & final)
        self.assertEqual(len(final), 30)
        self.assertEqual(min(final), 2000)
        self.assertEqual(max(final), 2029)

    def test_final_seed_guard(self):
        with self.assertRaises(ValueError):
            S.final_train_kwargs("adam", CFG, S.stage_settings(), 1999)


class ReevalJobTests(unittest.TestCase):
    def tops(self):
        return {opt: [{"rank": r, "trial": 10 * r + i, "value": 500.0, "config": dict(CFG, lr=1e-4 * (r + 1))}
                      for r in range(5)] for i, opt in enumerate(T.OPTIMIZERS)}

    def test_75_jobs_and_seeds(self):
        jobs = S.build_reeval_jobs("exp", self.tops(), S.stage_settings())
        self.assertEqual(len(jobs), 75)
        self.assertEqual(len({j["out_path"] for j in jobs}), 75)
        for opt in T.OPTIMIZERS:
            js = [j for j in jobs if j["opt"] == opt]
            self.assertEqual(len(js), 25)
            self.assertEqual(sorted(j["seed"] for j in js), list(range(1000, 1025)))
        j = jobs[0]["train_kwargs"]
        self.assertEqual((j["steps"], j["eval_every"], j["eval_episodes"]), (500_000, 500_000, 20))

    def test_same_seed_sets_across_optimizers(self):
        jobs = S.build_reeval_jobs("exp", self.tops(), S.stage_settings())
        by_opt = {opt: sorted((j["rank"], j["seed"]) for j in jobs if j["opt"] == opt) for opt in T.OPTIMIZERS}
        self.assertEqual(by_opt["adam"], by_opt["lion"])
        self.assertEqual(by_opt["adam"], by_opt["lpeco"])

    def test_jobs_are_interleaved(self):
        jobs = S.build_reeval_jobs("exp", self.tops(), S.stage_settings())
        self.assertEqual([j["opt"] for j in jobs[:3]], list(T.OPTIMIZERS))


class SelectionTests(unittest.TestCase):
    def test_highest_mean_wins(self):
        rows = [{"rank": 0, "trial": 5, "mean": 480.0, "values": []}, {"rank": 1, "trial": 9, "mean": 500.0, "values": []},
                {"rank": 2, "trial": 1, "mean": 450.0, "values": []}]
        self.assertEqual(S.select_after_reeval(rows)["trial"], 9)

    def test_exact_tie_goes_to_lower_trial_number(self):   # DEVIATIONS D1
        rows = [{"rank": 0, "trial": 22, "mean": 500.0, "values": []}, {"rank": 1, "trial": 5, "mean": 500.0, "values": []},
                {"rank": 2, "trial": 12, "mean": 500.0, "values": []}]
        self.assertEqual(S.select_after_reeval(rows)["trial"], 5)

    def test_variance_is_ignored(self):
        rows = [{"rank": 0, "trial": 3, "mean": 490.0, "values": [490.0] * 5},
                {"rank": 1, "trial": 8, "mean": 495.0, "values": [500.0, 500.0, 500.0, 500.0, 475.0]}]
        self.assertEqual(S.select_after_reeval(rows)["trial"], 8)


class FinalJobTests(unittest.TestCase):
    def selected(self):
        return {opt: {"config": dict(CFG, lr=lr)} for opt, lr in (("adam", 3e-4), ("lion", 2e-4), ("lpeco", 1.5e-4))}

    def test_120_jobs(self):
        jobs = S.build_final_jobs("exp", self.selected(), S.stage_settings())
        self.assertEqual(len(jobs), 120)
        self.assertEqual(len({j["out_path"] for j in jobs}), 120)
        for label in S.ALL_FINAL_LABELS:
            self.assertEqual(sorted(j["seed"] for j in jobs if j["label"] == label), list(range(2000, 2030)))
        kw = jobs[0]["train_kwargs"]
        self.assertEqual((kw["steps"], kw["eval_every"], kw["eval_episodes"]), (500_000, 25_000, 20))
        self.assertEqual(500_000 // kw["eval_every"], 20)

    def test_exploratory_is_lion_with_the_lpeco_config_and_cs_zero(self):
        jobs = S.build_final_jobs("exp", self.selected(), S.stage_settings())
        j = next(j for j in jobs if j["label"] == "lpeco_cs0" and j["seed"] == 2000)["train_kwargs"]
        self.assertEqual(j["opt"], "lion")
        self.assertEqual(j["cs"], 0.0)
        self.assertEqual(j["lr"], 1.5e-4)          # LPECO's tuned learning rate, not Lion's
        lp = next(j for j in jobs if j["label"] == "lpeco" and j["seed"] == 2000)["train_kwargs"]
        self.assertEqual(lp["opt"], "lpeco")
        self.assertEqual(lp["cs"], 0.2)

    def test_smoke_settings_give_20_checkpoints(self):
        s = S.stage_settings(smoke=True)
        self.assertEqual(s["steps"] // s["final_eval_every"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
