"""PyTorch port of LPECO (Lion with Predictive Error Correction).

The update rule is a line-by-line port of the Keras class
``LionPredictiveErrorCorrectionOptimizer`` in the original ``lpeco.py``
(and of Algorithm 1 in the paper). Per parameter, with gradient g and the
momentum m and error EMA I stored in the optimizer state:

    e  = g + m                         # uses the OLD momentum
    I  = gamma * I + (1 - gamma) * e   # error EMA ("integral" term)
    u  = beta1 * m + (1 - beta1) * g   # Lion interpolation
    u' = u - alpha * I                 # correction (alpha = correction_strength)
    w  = w - lr * sign(u')
    m  = beta2 * m + (1 - beta2) * g   # momentum update (AFTER the step)

With ``correction_strength=0`` the update is exactly Lion without weight
decay, so Lion in our experiments is simply ``LPECO(..., correction_strength=0)``.

Deliberate differences from a "full" Lion:
  * No weight decay (the protocol runs Lion without it).
  * ``error_decay`` must be < 1 (gamma = 1 would freeze the EMA at zero).

NOTE: this file has been checked against a numpy reference only through the
tests in ``test_lpeco_torch.py``; run them before using it for experiments.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


class LPECO(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        error_decay: float = 0.999,
        correction_strength: float = 0.1,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas (each must be in [0, 1)): {betas}")
        if not 0.0 <= error_decay < 1.0:
            raise ValueError(f"error_decay must be in [0, 1): {error_decay}")
        if not correction_strength >= 0.0:
            raise ValueError(f"correction_strength must be >= 0: {correction_strength}")
        defaults = dict(
            lr=lr,
            betas=betas,
            error_decay=error_decay,
            correction_strength=correction_strength,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            gamma = group["error_decay"]
            alpha = group["correction_strength"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LPECO does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["error_ema"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m = state["exp_avg"]
                error_ema = state["error_ema"]

                # prediction error uses the momentum BEFORE this step's update
                error = grad + m
                error_ema.mul_(gamma).add_(error, alpha=1.0 - gamma)

                interpolated = m * beta1 + grad * (1.0 - beta1)
                corrected = interpolated - alpha * error_ema
                p.add_(torch.sign(corrected), alpha=-lr)

                # momentum is updated last
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
                state["step"] += 1

        return loss
