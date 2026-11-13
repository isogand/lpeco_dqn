"""Pilot (PROTOCOL.md section 5.1): Adam, default configuration, seeds 9000-9004, 500k steps,
greedy evaluation every 25k steps. DESCRIPTIVE ONLY: it checks the pipeline and shows learning curves.
The horizon stays 500k whatever it shows, and pilot results are never used in the analysis.

    python3 pilot.py --smoke          # tiny rehearsal in ./pilot_smoke (about a minute)
    python3 pilot.py                  # the real pilot, ./pilot (about 7 minutes with 4 workers)

Resumable: finished runs are not recomputed (one JSON file per seed).
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

import dqn

PILOT = dict(
    opt="adam", lr=2.5e-4, target_every=500, steps=500_000, eval_every=25_000,
    eval_episodes=20, learning_starts=10_000, seeds=[9000, 9001, 9002, 9003, 9004], workers=4,
)
SMOKE = dict(PILOT, steps=4_000, eval_every=1_000, eval_episodes=3, learning_starts=500)

SATURATED, COLLAPSED = 490.0, 250.0  # same thresholds as the protocol's secondary metrics


def build_jobs(cfg, root):
    jobs = []
    for seed in cfg["seeds"]:
        assert 9000 <= seed <= 9004, "pilot seeds must stay in 9000-9004"
        jobs.append({
            "out_path": str(Path(root) / f"adam_seed{seed}.json"),
            "opt": cfg["opt"], "seed": seed,
            "train_kwargs": dict(
                opt=cfg["opt"], seed=seed, steps=cfg["steps"], lr=cfg["lr"], target_every=cfg["target_every"],
                cs=0.0, error_decay=0.995, beta1=0.9, beta2=0.99,
                eval_every=cfg["eval_every"], eval_episodes=cfg["eval_episodes"],
                learning_starts=cfg["learning_starts"],
            ),
        })
    return jobs


def collapses(evals):
    """Checkpoints after the first one with mean >= 490 whose mean is < 250 (protocol definition)."""
    first = next((i for i, (_, r) in enumerate(evals) if r >= SATURATED), None)
    if first is None:
        return None  # never reached the ceiling
    return sum(1 for _, r in evals[first + 1:] if r < COLLAPSED)


def summarize(results, cfg):
    steps = [s for s, _ in results[0]["evals"]]
    curves = np.array([[r for _, r in res["evals"]] for res in results])  # seeds x checkpoints
    print("\nDESCRIPTIVE SUMMARY (Adam, default config; not used in any analysis)")
    print(f"{'step':>8}  {'mean':>7}  {'min':>7}  {'max':>7}   per-seed greedy returns")
    for j, s in enumerate(steps):
        col = curves[:, j]
        print(f"{s:>8}  {col.mean():7.1f}  {col.min():7.1f}  {col.max():7.1f}   "
              + " ".join(f"{x:6.1f}" for x in col))
    print()
    for res in results:
        c = collapses(res["evals"])
        print(f"seed {res['seed']}: final {res['final_return']:.1f} | training mean(last 10 eps) "
              f"{res['mean_last10']:.1f} | collapses after reaching >=490: "
              f"{'never reached' if c is None else c} | hash {res['weights_hash']} | {res['wall_s']:.0f}s")
    finals = curves[:, -1]
    out = {
        "config": {k: v for k, v in cfg.items()},
        "checkpoint_steps": steps,
        "mean_curve": curves.mean(axis=0).tolist(),
        "final_returns": finals.tolist(),
        "collapses": [collapses(r["evals"]) for r in results],
        "note": "descriptive only; never used in the analysis",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else PILOT
    root = args.root or ("pilot_smoke" if args.smoke else "pilot")
    Path(root).mkdir(parents=True, exist_ok=True)
    print(f"pilot: {cfg['opt']} lr={cfg['lr']} target_every={cfg['target_every']} steps={cfg['steps']:,} "
          f"eval every {cfg['eval_every']:,} x {cfg['eval_episodes']} episodes, seeds {cfg['seeds']}")

    jobs = build_jobs(cfg, root)
    ctx = mp.get_context("spawn")
    pool = cf.ProcessPoolExecutor(max_workers=cfg["workers"], mp_context=ctx)
    try:
        futures = [pool.submit(dqn.run_job, j) for j in jobs]
        results = [f.result() for f in futures]
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("Interrupted. Run the same command again to resume; finished seeds are kept.")
        sys.exit(130)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    summary = summarize(results, cfg)
    (Path(root) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten: {root}/summary.json and one JSON per seed")


if __name__ == "__main__":
    main()
