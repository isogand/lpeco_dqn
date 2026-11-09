"""Small DQN on CartPole for timing and reproducibility checks.

This is a FRESH script (not your old time_dqn.py). It is meant to answer three
questions before the real experiments:
  1. Does the same --seed give exactly the same run?   (compare the "weights" hash)
  2. Does periodic greedy evaluation leave training untouched?
     (run with and without --eval-every; the hash must be identical)
  3. How long does a run take with adam / lion / lpeco?

Examples (run inside the lpeco_dqn folder):
    python3 time_dqn.py --steps 50000 --seed 1
    python3 time_dqn.py --steps 50000 --seed 1 --eval-every 10000
    python3 time_dqn.py --steps 50000 --seed 1 --opt lpeco --cs 0.2
    python3 time_dqn.py --steps 50000 --seed 1 --opt lion        # = LPECO with cs=0

The hyperparameter defaults below are PLACEHOLDERS (written from memory of the
CleanRL CartPole DQN defaults, not copied from a file). They are not the
locked values of the experiment: the final values must be written in the
protocol table.
"""
import argparse
import hashlib
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


def make_q(n_obs, n_act):
    return torch.nn.Sequential(
        torch.nn.Linear(n_obs, 120), torch.nn.ReLU(),
        torch.nn.Linear(120, 84), torch.nn.ReLU(),
        torch.nn.Linear(84, n_act),
    )


def make_optimizer(name, params, lr, betas, error_decay, cs):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    from lpeco_torch import LPECO  # imported only when needed

    if name == "lion":  # Lion == LPECO with correction_strength = 0
        return LPECO(params, lr=lr, betas=betas, error_decay=error_decay, correction_strength=0.0)
    if name == "lpeco":
        return LPECO(params, lr=lr, betas=betas, error_decay=error_decay, correction_strength=cs)
    raise ValueError(f"unknown optimizer: {name}")


@torch.no_grad()
def evaluate(q, env, base_seed, episodes):
    """Greedy (epsilon = 0) evaluation on a SEPARATE env with fixed seeds.

    Uses no training RNG (no numpy/torch random calls), so calling it cannot
    change the training trajectory.
    """
    rets = []
    for i in range(episodes):
        obs, _ = env.reset(seed=base_seed + i)
        done, ep_ret = False, 0.0
        while not done:
            a = int(q(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax(1).item())
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += r
            done = term or trunc
        rets.append(ep_ret)
    return float(np.mean(rets))


def weights_hash(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", choices=["adam", "lion", "lpeco"], default="adam")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--cs", type=float, default=0.2, help="LPECO correction_strength")
    ap.add_argument("--error-decay", type=float, default=0.995)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--eval-every", type=int, default=0, help="0 = no periodic evaluation")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    # placeholder DQN hyperparameters (see module docstring)
    ap.add_argument("--buffer", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target-every", type=int, default=500)
    ap.add_argument("--train-every", type=int, default=10)
    ap.add_argument("--learning-starts", type=int, default=10_000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-fraction", type=float, default=0.5)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)  # ALL training randomness (epsilon, replay sampling)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")  # separate env for evaluation
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n
    EVAL_SEED_BASE = 10_000_000 + a.seed * 1000  # never overlaps training seeds

    q, q_t = make_q(n_obs, n_act), make_q(n_obs, n_act)
    q_t.load_state_dict(q.state_dict())
    opt = make_optimizer(a.opt, q.parameters(), a.lr, (a.beta1, a.beta2), a.error_decay, a.cs)

    N = a.buffer
    B = {
        "o": np.zeros((N, n_obs), np.float32), "o2": np.zeros((N, n_obs), np.float32),
        "a": np.zeros(N, np.int64), "r": np.zeros(N, np.float32), "d": np.zeros(N, np.float32),
    }
    filled = 0

    obs, _ = env.reset(seed=a.seed)
    ep_ret, rets, evals = 0.0, [], []
    t_learn = None
    t0 = time.perf_counter()

    for t in range(a.steps):
        frac = min(1.0, t / max(1.0, a.eps_fraction * a.steps))
        eps = a.eps_start + frac * (a.eps_end - a.eps_start)
        if rng.random() < eps:
            act = int(rng.integers(n_act))
        else:
            with torch.no_grad():
                act = int(q(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax(1).item())

        obs2, r, term, trunc, _ = env.step(act)
        i = t % N
        B["o"][i], B["o2"][i], B["a"][i], B["r"][i] = obs, obs2, act, r
        B["d"][i] = float(term)  # truncation is NOT a terminal state
        filled = min(filled + 1, N)
        ep_ret += r
        if term or trunc:
            rets.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        else:
            obs = obs2

        if t >= a.learning_starts:
            if t_learn is None:
                t_learn = time.perf_counter()
            if t % a.train_every == 0:
                idx = rng.integers(0, filled, a.batch)
                ob, o2b = torch.as_tensor(B["o"][idx]), torch.as_tensor(B["o2"][idx])
                ab, rb, db = (torch.as_tensor(B[k][idx]) for k in ("a", "r", "d"))
                with torch.no_grad():
                    tgt = rb + a.gamma * q_t(o2b).max(1).values * (1 - db)
                loss = F.mse_loss(q(ob).gather(1, ab[:, None]).squeeze(1), tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if t % a.target_every == 0:
                q_t.load_state_dict(q.state_dict())

        if a.eval_every and (t + 1) % a.eval_every == 0:
            evals.append((t + 1, evaluate(q, eval_env, EVAL_SEED_BASE, a.eval_episodes)))

    tot = time.perf_counter() - t0
    print(f"opt={a.opt} seed={a.seed} steps={a.steps}")
    print(f"total: {tot:.1f}s, overall SPS: {a.steps / tot:.0f}")
    if t_learn:
        print(f"after learning starts: SPS = {(a.steps - a.learning_starts) / (time.perf_counter() - t_learn):.0f}")
    print(f"mean return of last 10 episodes: {np.mean(rets[-10:]):.1f}")
    for step, ret in evals:
        print(f"  greedy eval @ {step}: {ret:.1f}")
    print(f"weights: {weights_hash(q)}")


if __name__ == "__main__":
    main()
