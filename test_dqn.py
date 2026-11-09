"""Regression tests for dqn.py (needs torch, gymnasium, lpeco_torch.py).

    python3 -m unittest -v test_dqn          # takes about 1-2 minutes (several 50k-step runs)

The expected hashes are the ones you measured with time_dqn.py on THIS machine/software stack
(torch 2.2.2, numpy 1.26.4, gymnasium 1.0.0). They are not expected to match on a different machine.
"""
import unittest

from dqn import train

STEPS = 50_000


class RegressionAgainstTimeDqn(unittest.TestCase):
    def test_adam_seed1(self):
        r = train(opt="adam", seed=1, steps=STEPS)
        self.assertEqual(r["weights_hash"], "7e5525305cd54aa7")
        self.assertEqual(round(r["mean_last10"], 1), 237.6)

    def test_lion_seed1(self):
        r = train(opt="lion", seed=1, steps=STEPS)
        self.assertEqual(r["weights_hash"], "8ee8a92d62294f8a")
        self.assertEqual(round(r["mean_last10"], 1), 145.4)

    def test_lpeco_cs0_equals_lion(self):
        r = train(opt="lpeco", seed=1, steps=STEPS, cs=0.0)
        self.assertEqual(r["weights_hash"], "8ee8a92d62294f8a")

    def test_lpeco_cs02_seed1(self):
        r = train(opt="lpeco", seed=1, steps=STEPS, cs=0.2)
        self.assertEqual(r["weights_hash"], "b07ec9705a126cd6")
        self.assertEqual(round(r["mean_last10"], 1), 357.8)


class EvaluationInvariance(unittest.TestCase):
    def test_periodic_eval_does_not_change_training(self):
        plain = train(opt="adam", seed=1, steps=STEPS)
        with_eval = train(opt="adam", seed=1, steps=STEPS, eval_every=10_000, eval_episodes=5)
        self.assertEqual(plain["weights_hash"], with_eval["weights_hash"])
        self.assertEqual(len(with_eval["evals"]), 5)
        self.assertEqual(with_eval["final_return"], with_eval["evals"][-1][1])

    def test_final_return_is_none_without_final_eval(self):
        r = train(opt="adam", seed=2, steps=3000, learning_starts=500, eval_every=0)
        self.assertIsNone(r["final_return"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
