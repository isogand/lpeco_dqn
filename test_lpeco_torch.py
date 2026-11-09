"""Tests for lpeco_torch.LPECO.

Run from the folder containing lpeco_torch.py:

    python -m unittest -v test_lpeco_torch

Only torch + numpy are required. KerasReferenceTest additionally needs
tensorflow/keras and lpeco_keras_ref.py (verbatim copy of the original Keras
class); it is skipped automatically if they are missing.

Idea: the numpy functions below are an independent re-implementation of the
paper's Algorithm 1. If torch == numpy reference and (optionally)
Keras == numpy reference, then torch == Keras by transitivity.
"""
import copy
import unittest

import numpy as np
import torch

from lpeco_torch import LPECO


# --------------------------------------------------------------------------
# Independent numpy references
# --------------------------------------------------------------------------
def lpeco_ref_step(w, g, m, I, lr, b1, b2, gamma, alpha):
    e = g + m                                # error uses OLD momentum
    I = gamma * I + (1.0 - gamma) * e        # error EMA
    u = b1 * m + (1.0 - b1) * g              # Lion interpolation
    d = np.sign(u - alpha * I)               # corrected direction
    w = w - lr * d
    m = b2 * m + (1.0 - b2) * g              # momentum after the step
    return w, m, I


def lion_ref_step(w, g, m, lr, b1, b2):
    c = b1 * m + (1.0 - b1) * g
    w = w - lr * np.sign(c)
    m = b2 * m + (1.0 - b2) * g
    return w, m


def run_torch(w0, grads, **kw):
    """Run LPECO on one float64 parameter, feeding the given gradient list."""
    p = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    opt = LPECO([p], **kw)
    for g in grads:
        p.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
    return p.detach().numpy().copy()


def make_grads(rng, shape, n, drift=True):
    """Noisy gradients whose scale changes over time (like a training loss)."""
    grads = []
    for t in range(n):
        scale = (1.0 + 2.0 * np.sin(t / 25.0)) if drift else 1.0
        grads.append(rng.normal(size=shape) * scale + 0.3 * rng.normal(size=shape))
    return grads


# --------------------------------------------------------------------------
class NumpyReferenceTest(unittest.TestCase):
    """torch implementation == independent numpy implementation."""

    def check(self, shape, lr, b1, b2, gamma, alpha, n=300, seed=0):
        rng = np.random.default_rng(seed)
        w0 = rng.normal(size=shape)
        grads = make_grads(rng, shape, n)

        w, m, I = w0.copy(), np.zeros(shape), np.zeros(shape)
        for g in grads:
            w, m, I = lpeco_ref_step(w, g, m, I, lr, b1, b2, gamma, alpha)

        w_torch = run_torch(
            w0, grads, lr=lr, betas=(b1, b2), error_decay=gamma, correction_strength=alpha
        )
        # One sign flip would move a weight by 2*lr (>= 2e-4 here), far above atol.
        np.testing.assert_allclose(w_torch, w, rtol=0, atol=1e-10)

    def test_vector_default_like(self):
        self.check((7,), lr=1e-3, b1=0.9, b2=0.99, gamma=0.999, alpha=0.1)

    def test_matrix_champion_like(self):
        self.check((3, 4), lr=6e-4, b1=0.9, b2=0.99, gamma=0.999, alpha=0.3, seed=1)

    def test_other_hyperparameters(self):
        self.check((5,), lr=3e-4, b1=0.95, b2=0.999, gamma=0.99, alpha=0.35, seed=2)

    def test_upper_bound_alpha(self):
        self.check((6,), lr=1e-3, b1=0.9, b2=0.99, gamma=0.995, alpha=0.4, seed=3)


class LionEquivalenceTest(unittest.TestCase):
    """correction_strength = 0  =>  exactly Lion without weight decay."""

    def test_cs_zero_equals_lion_reference(self):
        rng = np.random.default_rng(10)
        shape = (9,)
        w0 = rng.normal(size=shape)
        grads = make_grads(rng, shape, 400)
        lr, b1, b2 = 1e-3, 0.9, 0.99

        w, m = w0.copy(), np.zeros(shape)
        for g in grads:
            w, m = lion_ref_step(w, g, m, lr, b1, b2)

        w_torch = run_torch(
            w0, grads, lr=lr, betas=(b1, b2), error_decay=0.999, correction_strength=0.0
        )
        np.testing.assert_allclose(w_torch, w, rtol=0, atol=1e-12)

    def test_cs_zero_independent_of_error_decay(self):
        rng = np.random.default_rng(11)
        w0 = rng.normal(size=(4,))
        grads = make_grads(rng, (4,), 200)
        a = run_torch(w0, grads, lr=1e-3, error_decay=0.9, correction_strength=0.0)
        b = run_torch(w0, grads, lr=1e-3, error_decay=0.999, correction_strength=0.0)
        np.testing.assert_array_equal(a, b)


class HandComputedTest(unittest.TestCase):
    """Two steps worked out by hand (lr=0.01, b1=0.9, b2=0.99, gamma=0.999, alpha=0.1, g=1, w0=1).

    step 1: e=1,     I=0.001,    u=0.1,   u'=0.0999  -> w=0.99, m=0.01
    step 2: e=1.01,  I=0.002009, u=0.109, u'=0.10880 -> w=0.98, m=0.0199
    """

    def test_two_steps(self):
        p = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        opt = LPECO([p], lr=0.01, betas=(0.9, 0.99), error_decay=0.999, correction_strength=0.1)
        for _ in range(2):
            p.grad = torch.tensor([1.0], dtype=torch.float64)
            opt.step()
        st = opt.state[p]
        self.assertAlmostEqual(p.item(), 0.98, places=12)
        self.assertAlmostEqual(st["exp_avg"].item(), 0.0199, places=12)
        self.assertAlmostEqual(st["error_ema"].item(), 0.002009, places=12)
        self.assertEqual(st["step"], 2)

    def test_zero_gradient_first_step_does_not_move(self):
        p = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        opt = LPECO([p], lr=0.1, correction_strength=0.3)
        p.grad = torch.zeros(1, dtype=torch.float64)
        opt.step()
        self.assertEqual(p.item(), 1.5)  # sign(0) == 0


class SteadyStateTest(unittest.TestCase):
    """With a constant gradient the correction acts as (1 - 2*alpha) * g:
    alpha < 0.5 keeps the descent direction, alpha > 0.5 reverses it.
    (This is why the experiments cap correction_strength at 0.4.)"""

    def last_step_direction(self, alpha):
        p = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        opt = LPECO([p], lr=1e-3, betas=(0.9, 0.9), error_decay=0.99, correction_strength=alpha)
        prev = p.item()
        for _ in range(3000):
            p.grad = torch.tensor([1.0], dtype=torch.float64)
            opt.step()
            cur = p.item()
            delta, prev = cur - prev, cur
        return delta

    def test_alpha_below_half_descends(self):
        self.assertLess(self.last_step_direction(0.3), 0.0)

    def test_alpha_above_half_reverses(self):
        self.assertGreater(self.last_step_direction(0.6), 0.0)

    def test_steady_state_error_ema_is_twice_gradient(self):
        p = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        opt = LPECO([p], lr=1e-3, betas=(0.9, 0.9), error_decay=0.99, correction_strength=0.2)
        for _ in range(3000):
            p.grad = torch.tensor([1.0], dtype=torch.float64)
            opt.step()
        self.assertAlmostEqual(opt.state[p]["exp_avg"].item(), 1.0, places=6)
        self.assertAlmostEqual(opt.state[p]["error_ema"].item(), 2.0, places=6)


class StateAndApiTest(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        rng = np.random.default_rng(20)
        w0 = rng.normal(size=(5,))
        grads = make_grads(rng, (5,), 60)
        kw = dict(lr=1e-3, betas=(0.9, 0.99), error_decay=0.995, correction_strength=0.2)

        full = run_torch(w0, grads, **kw)

        p1 = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
        o1 = LPECO([p1], **kw)
        for g in grads[:30]:
            p1.grad = torch.tensor(g, dtype=torch.float64)
            o1.step()
        saved = copy.deepcopy(o1.state_dict())

        p2 = torch.tensor(p1.detach().numpy().copy(), dtype=torch.float64, requires_grad=True)
        o2 = LPECO([p2], **kw)
        o2.load_state_dict(saved)
        for g in grads[30:]:
            p2.grad = torch.tensor(g, dtype=torch.float64)
            o2.step()
        np.testing.assert_allclose(p2.detach().numpy(), full, rtol=0, atol=1e-12)

    def test_param_without_grad_is_skipped(self):
        a = torch.tensor([1.0], requires_grad=True)
        b = torch.tensor([2.0], requires_grad=True)
        opt = LPECO([a, b], lr=0.1)
        a.grad = torch.tensor([1.0])
        opt.step()
        self.assertEqual(b.item(), 2.0)
        self.assertNotIn(b, opt.state)

    def test_closure_returns_loss(self):
        p = torch.tensor([1.0], requires_grad=True)
        opt = LPECO([p], lr=0.01)

        def closure():
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            return loss

        loss = opt.step(closure)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        self.assertLess(p.item(), 1.0)

    def test_invalid_hyperparameters(self):
        p = [torch.zeros(1, requires_grad=True)]
        with self.assertRaises(ValueError):
            LPECO(p, lr=-1.0)
        with self.assertRaises(ValueError):
            LPECO(p, betas=(1.0, 0.99))
        with self.assertRaises(ValueError):
            LPECO(p, error_decay=1.0)
        with self.assertRaises(ValueError):
            LPECO(p, correction_strength=-0.1)

    def test_float32_smoke_training(self):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
        x = torch.randn(64, 4)
        y = (x[:, 0] > 0).long()
        opt = LPECO(net.parameters(), lr=1e-2, correction_strength=0.2)
        first = None
        for _ in range(200):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(x), y)
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        self.assertTrue(np.isfinite(loss.item()))
        self.assertLess(loss.item(), first)


# --------------------------------------------------------------------------
# Optional: the ORIGINAL Keras class vs the numpy reference
# (run this once on your Mac; I could not run it myself)
# --------------------------------------------------------------------------
try:
    import keras
    import tensorflow as tf
    from lpeco_keras_ref import LionPredictiveErrorCorrectionOptimizer

    HAVE_KERAS = True
except Exception:  # pragma: no cover
    HAVE_KERAS = False


@unittest.skipUnless(HAVE_KERAS, "tensorflow/keras or lpeco_keras_ref.py not available")
class KerasReferenceTest(unittest.TestCase):
    def check(self, alpha, seed):
        rng = np.random.default_rng(seed)
        shape = (6,)
        lr, b1, b2, gamma = 1e-3, 0.9, 0.99, 0.995
        w0 = rng.normal(size=shape).astype("float32")
        grads = [g.astype("float32") for g in make_grads(rng, shape, 120)]

        w, m, I = w0.astype("float64"), np.zeros(shape), np.zeros(shape)
        for g in grads:
            w, m, I = lpeco_ref_step(w, g.astype("float64"), m, I, lr, b1, b2, gamma, alpha)

        var = keras.Variable(w0.copy())
        opt = LionPredictiveErrorCorrectionOptimizer(
            learning_rate=lr, beta_1=b1, beta_2=b2, error_decay=gamma, correction_strength=alpha
        )
        for g in grads:
            opt.apply_gradients([(tf.constant(g), var)])
        w_keras = np.array(var.numpy(), dtype="float64")
        # float32 vs float64 rounding is ~1e-7; a sign flip would move a weight by 2e-3.
        np.testing.assert_allclose(w_keras, w, rtol=0, atol=1e-5)

    def test_keras_matches_numpy_alpha_03(self):
        self.check(alpha=0.3, seed=0)

    def test_keras_matches_numpy_alpha_01(self):
        self.check(alpha=0.1, seed=1)

    def test_keras_cs_zero_matches_lion_reference(self):
        self.check(alpha=0.0, seed=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
