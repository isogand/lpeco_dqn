"""Pure logic for the re-evaluation and final stages (PROTOCOL.md sections 5.3-5.4, DEVIATIONS.md D1).

No torch / optuna imports: the selection works only from the per-run JSON files written by tune.py.
"""
import json
from pathlib import Path
from statistics import mean

import tune_lib as T

TOPK = 5
REEVAL_SEEDS_PER_CONFIG = 5
REEVAL_SEED_RANGE = (1000, 1999)
FINAL_SEEDS = tuple(range(2000, 2030))
FINAL_SEED_RANGE = (2000, 2999)
FINAL_LABELS = ("adam", "lion", "lpeco")
EXPLORATORY_LABEL = "lpeco_cs0"   # best LPECO config re-run with correction_strength = 0 (exploratory)
ALL_FINAL_LABELS = FINAL_LABELS + (EXPLORATORY_LABEL,)


def stage_settings(smoke=False):
    """Run settings for the re-evaluation and final stages (smoke = tiny rehearsal, never real results)."""
    if smoke:
        return dict(steps=3_000, learning_starts=500, eval_episodes=3, final_eval_every=150,
                    final_seeds=tuple(range(2000, 2006)), workers=4)
    return dict(steps=T.PROTOCOL["steps"], learning_starts=T.PROTOCOL["learning_starts"],
                eval_episodes=T.PROTOCOL["eval_episodes"], final_eval_every=25_000,
                final_seeds=FINAL_SEEDS, workers=T.PROTOCOL["workers"])


# ---------------------------------------------------------------------------
# Tie-break (PROTOCOL 5.3 and DEVIATIONS D1): higher value first, then LOWER tuning trial number
# ---------------------------------------------------------------------------
def tie_key(value, trial):
    return (-round(float(value), 9), int(trial))


def load_tuning_trials(root, opt, seeds_per_trial=3):
    """{trial_number: {"seeds", "values", "value", "config"}} read from root/tuning/<opt>/*.json."""
    trials = {}
    for f in sorted((Path(root) / "tuning" / opt).glob("trial*_seed*.json")):
        r = json.loads(f.read_text())
        k = int(r["job"]["trial"])
        t = trials.setdefault(k, {"seeds": [], "values": [], "config": r["config"]})
        t["seeds"].append(int(r["seed"]))
        t["values"].append(float(r["final_return"]))
    bad = {k: len(v["values"]) for k, v in trials.items() if len(v["values"]) != seeds_per_trial}
    if bad:
        raise ValueError(f"{opt}: trials with an incomplete set of seeds: {bad}")
    for t in trials.values():
        t["value"] = mean(t["values"])
    return trials


def top_k(trials, k=TOPK):
    """Top-k trials by tuning objective; exact ties go to the lower trial number."""
    ranked = sorted(trials.items(), key=lambda kv: tie_key(kv[1]["value"], kv[0]))
    return [{"rank": i, "trial": n, "value": t["value"], "config": t["config"]}
            for i, (n, t) in enumerate(ranked[:k])]


# ---------------------------------------------------------------------------
# Re-evaluation jobs
# ---------------------------------------------------------------------------
def reeval_seeds(rank):
    seeds = [REEVAL_SEED_RANGE[0] + REEVAL_SEEDS_PER_CONFIG * rank + i for i in range(REEVAL_SEEDS_PER_CONFIG)]
    if min(seeds) < REEVAL_SEED_RANGE[0] or max(seeds) > REEVAL_SEED_RANGE[1]:
        raise ValueError("re-evaluation seeds out of range")
    return seeds


def _train_kwargs(opt, seed, cfg, settings, eval_every, cs=None):
    return dict(
        opt=opt, seed=seed, steps=settings["steps"],
        lr=float(cfg["lr"]), target_every=int(cfg["target_every"]),
        cs=float(cfg["cs"] if cs is None else cs), error_decay=float(cfg["error_decay"]),
        beta1=float(cfg["beta1"]), beta2=float(cfg["beta2"]),
        eval_every=eval_every, eval_episodes=settings["eval_episodes"],
        learning_starts=settings["learning_starts"],
    )


def build_reeval_jobs(root, tops, settings):
    """Interleaved list of jobs: every optimizer gets rank 0 first, then rank 1, ..."""
    jobs = []
    for rank in range(TOPK):
        for i in range(REEVAL_SEEDS_PER_CONFIG):
            for opt in T.OPTIMIZERS:
                entry = tops[opt][rank]
                seed = reeval_seeds(rank)[i]
                out = Path(root) / "reeval" / opt / f"rank{rank}_trial{entry['trial']:03d}_seed{seed:04d}.json"
                jobs.append({
                    "out_path": str(out), "opt": opt, "rank": rank, "trial": entry["trial"], "seed": seed,
                    "train_kwargs": _train_kwargs(opt, seed, entry["config"], settings, settings["steps"]),
                })
    return jobs


def select_after_reeval(rows):
    """rows: [{"rank","trial","mean","values"}]. Highest re-evaluation mean; exact ties -> lower trial (D1)."""
    return min(rows, key=lambda r: tie_key(r["mean"], r["trial"]))


def load_reeval_rows(root, opt):
    groups = {}
    for f in sorted((Path(root) / "reeval" / opt).glob("rank*_trial*_seed*.json")):
        r = json.loads(f.read_text())
        g = groups.setdefault(int(r["job"]["rank"]), {"rank": int(r["job"]["rank"]), "trial": int(r["job"]["trial"]),
                                                      "values": [], "config": r["config"]})
        g["values"].append(float(r["final_return"]))
    rows = []
    for g in groups.values():
        if len(g["values"]) != REEVAL_SEEDS_PER_CONFIG:
            raise ValueError(f"{opt} rank {g['rank']}: {len(g['values'])} results, expected {REEVAL_SEEDS_PER_CONFIG}")
        g["mean"] = mean(g["values"])
        rows.append(g)
    return sorted(rows, key=lambda r: r["rank"])


# ---------------------------------------------------------------------------
# Final-evaluation jobs
# ---------------------------------------------------------------------------
def final_train_kwargs(label, cfg, settings, seed):
    if not (FINAL_SEED_RANGE[0] <= seed <= FINAL_SEED_RANGE[1]):
        raise ValueError(f"final seed {seed} outside {FINAL_SEED_RANGE}")
    if label == EXPLORATORY_LABEL:   # Lion == LPECO with cs = 0, using the selected LPECO configuration
        return _train_kwargs("lion", seed, cfg, settings, settings["final_eval_every"], cs=0.0)
    return _train_kwargs(label, seed, cfg, settings, settings["final_eval_every"])


def build_final_jobs(root, selected, settings):
    """selected: {opt: {"config": ...}} from selected_configs.json. Interleaved by seed."""
    jobs = []
    for seed in settings["final_seeds"]:
        for label in ALL_FINAL_LABELS:
            cfg = selected["lpeco" if label == EXPLORATORY_LABEL else label]["config"]
            out = Path(root) / "final" / label / f"seed{seed:04d}.json"
            jobs.append({"out_path": str(out), "label": label, "seed": seed,
                         "train_kwargs": final_train_kwargs(label, cfg, settings, seed)})
    return jobs
