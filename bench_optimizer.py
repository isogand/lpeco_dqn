"""Micro-benchmark: cost of one DQN-style update (forward + backward + optimizer step)
for adam / lion / lpeco, using the same network and batch size as time_dqn.py.

Run inside the lpeco_dqn folder (needs time_dqn.py and lpeco_torch.py next to it):

    python3 bench_optimizer.py

It prints, per optimizer, the best-of-3 time per update and the time spent only
inside opt.step(), then estimates the extra cost per 500k-step run relative to adam
(updates per run = (steps - learning_starts) / train_every, with time_dqn.py defaults).
"""
import time

import torch
import torch.nn.functional as F

from time_dqn import make_optimizer, make_q

BATCH = 128
STEPS, LEARNING_STARTS, TRAIN_EVERY = 500_000, 10_000, 10
N_UPDATES_PER_RUN = (STEPS - LEARNING_STARTS) // TRAIN_EVERY


def bench(name, n=3000, warmup=300):
    torch.manual_seed(0)
    q = make_q(4, 2)
    opt = make_optimizer(name, q.parameters(), 1e-4, (0.9, 0.99), 0.995, 0.2)
    x = torch.randn(BATCH, 4)
    target = torch.randn(BATCH)

    def one_update(time_step_only=False):
        opt.zero_grad()
        loss = F.mse_loss(q(x).max(1).values, target)
        loss.backward()
        if time_step_only:
            t = time.perf_counter()
            opt.step()
            return time.perf_counter() - t
        opt.step()
        return 0.0

    for _ in range(warmup):
        one_update()
    t0 = time.perf_counter()
    step_only = 0.0
    for _ in range(n):
        step_only += one_update(time_step_only=True)
    total = time.perf_counter() - t0
    return total / n * 1e3, step_only / n * 1e3  # ms per update, ms in opt.step


def main():
    torch.set_num_threads(1)
    results = {}
    for name in ("adam", "lion", "lpeco"):
        runs = [bench(name) for _ in range(3)]
        results[name] = min(r[0] for r in runs), min(r[1] for r in runs)
        print(f"{name:6s} total {results[name][0]:.3f} ms/update   opt.step {results[name][1]:.3f} ms")
    base = results["adam"][0]
    print(f"\nupdates per {STEPS:,}-step run: {N_UPDATES_PER_RUN:,}")
    for name in ("lion", "lpeco"):
        extra = (results[name][0] - base) * N_UPDATES_PER_RUN / 1e3
        print(f"{name:6s} extra time per run vs adam: {extra:+.1f} s")


if __name__ == "__main__":
    main()
