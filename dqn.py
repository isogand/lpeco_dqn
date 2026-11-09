"""DQN on CartPole as an importable function (refactor of time_dqn.py, same training logic).

`train(...)` must reproduce time_dqn.py exactly for the same arguments: the regression test
`test_dqn.py` checks the final-weights hashes you measured with time_dqn.py.

`run_job(job)` runs one training job and writes its result as JSON, atomically; if the result file
already exists it is returned without recomputing (this is what makes the tuning resumable).
"""
import hashlib
import json
import os
import random
import time
from pathlib import Path

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
    """Greedy (epsilon = 0) evaluation on a SEPARATE env with fixed seeds; uses no training RNG."""
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


def train(
    opt="adam",
    seed=1,
    steps=50_000,
    lr=2.5e-4,
    target_every=500,
    cs=0.2,
    error_decay=0.995,
    beta1=0.9,
    beta2=0.99,
    eval_every=0,
    eval_episodes=20,
    threads=1,
    buffer=10_000,
    batch=128,
    gamma=0.99,
    train_every=10,
    learning_starts=10_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_fraction=0.5,
):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)  # ALL training randomness (epsilon, replay sampling)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")  # separate env for evaluation
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n
    EVAL_SEED_BASE = 10_000_000 + seed * 1000  # never overlaps training seeds

    q, q_t = make_q(n_obs, n_act), make_q(n_obs, n_act)
    q_t.load_state_dict(q.state_dict())
    optimizer = make_optimizer(opt, q.parameters(), lr, (beta1, beta2), error_decay, cs)

    N = buffer
    B = {
        "o": np.zeros((N, n_obs), np.float32), "o2": np.zeros((N, n_obs), np.float32),
        "a": np.zeros(N, np.int64), "r": np.zeros(N, np.float32), "d": np.zeros(N, np.float32),
    }
    filled = 0

    obs, _ = env.reset(seed=seed)
    ep_ret, rets, evals = 0.0, [], []
    t_learn = None
    t0 = time.perf_counter()

    for t in range(steps):
        frac = min(1.0, t / max(1.0, eps_fraction * steps))
        eps = eps_start + frac * (eps_end - eps_start)
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

        if t >= learning_starts:
            if t_learn is None:
                t_learn = time.perf_counter()
            if t % train_every == 0:
                idx = rng.integers(0, filled, batch)
                ob, o2b = torch.as_tensor(B["o"][idx]), torch.as_tensor(B["o2"][idx])
                ab, rb, db = (torch.as_tensor(B[k][idx]) for k in ("a", "r", "d"))
                with torch.no_grad():
                    tgt = rb + gamma * q_t(o2b).max(1).values * (1 - db)
                loss = F.mse_loss(q(ob).gather(1, ab[:, None]).squeeze(1), tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if t % target_every == 0:
                q_t.load_state_dict(q.state_dict())

        if eval_every and (t + 1) % eval_every == 0:
            evals.append((t + 1, evaluate(q, eval_env, EVAL_SEED_BASE, eval_episodes)))

    tot = time.perf_counter() - t0
    sps_after = None
    if t_learn:
        sps_after = (steps - learning_starts) / (time.perf_counter() - t_learn)
    final_return = next((ret for step, ret in evals if step == steps), None)
    return {
        "opt": opt,
        "seed": seed,
        "steps": steps,
        "config": {
            "lr": lr, "target_every": target_every, "cs": cs, "error_decay": error_decay,
            "beta1": beta1, "beta2": beta2, "buffer": buffer, "batch": batch, "gamma": gamma,
            "train_every": train_every, "learning_starts": learning_starts,
            "eps_start": eps_start, "eps_end": eps_end, "eps_fraction": eps_fraction,
            "eval_every": eval_every, "eval_episodes": eval_episodes,
        },
        "final_return": final_return,
        "evals": [[s, r] for s, r in evals],
        "mean_last10": float(np.mean(rets[-10:])) if rets else None,
        "weights_hash": weights_hash(q),
        "wall_s": tot,
        "sps_after_learning": sps_after,
    }


def run_job(job):
    """Run one job dict (see tune_lib.build_job); skip the work if its result file already exists."""
    out = Path(job["out_path"])
    if out.exists():
        try:
            return json.loads(out.read_text())
        except Exception:
            pass  # unreadable/partial file: recompute
    res = train(**job["train_kwargs"])
    res["job"] = {k: v for k, v in job.items() if k != "train_kwargs"}
    res["finished_at"] = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)  # atomic: a crash never leaves a half-written result
    return res
