"""Pure logic for the tuning stage (no torch / optuna imports, so it is unit-testable anywhere).

Everything here mirrors PROTOCOL.md (v1 draft rev 2). If you change a constant, that is a protocol
deviation: write it in DEVIATIONS.md first.
"""
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants from PROTOCOL.md
# ---------------------------------------------------------------------------
PROTOCOL = dict(
    steps=500_000,
    learning_starts=10_000,
    eval_episodes=20,
    phase1_trials=48,
    phase2_trials=64,
    seeds_per_trial=3,
    check_after_active_s=2 * 3600,   # the single throughput check, after 2 h of ACTIVE time
    min_runs_at_check=130,           # runs that must be complete at that moment to allow phase 2
    gap_s=300,                       # a gap > 5 min with no completed run is not counted as active time
    workers=4,
    sampler_seed=0,
    n_startup_trials=10,
    beta1=0.9,
    beta2=0.99,
    unused_error_decay=0.995,        # passed to Adam/Lion runs, where it has no effect
    max_tuning_seed=999,             # tuning seeds must stay in 0..999
)

# Tiny settings used ONLY by `tune.py --smoke` (never for real results).
SMOKE = dict(PROTOCOL)
SMOKE.update(
    steps=3_000,
    learning_starts=500,
    eval_episodes=3,
    phase1_trials=3,
    phase2_trials=5,
    check_after_active_s=15,
    min_runs_at_check=6,
    n_startup_trials=2,
)

OPTIMIZERS = ("adam", "lion", "lpeco")

# name -> ("float", low, high, log) or ("cat", [choices]); insertion order = suggestion order
TARGET_FREQS = [250, 500, 1000, 2000]
SPACES = {
    "adam": {
        "lr": ("float", 5e-5, 2e-3, True),
        "target_every": ("cat", TARGET_FREQS),
    },
    "lion": {
        "lr": ("float", 1e-5, 1e-3, True),
        "target_every": ("cat", TARGET_FREQS),
    },
    "lpeco": {
        "lr": ("float", 1e-5, 1e-3, True),
        "target_every": ("cat", TARGET_FREQS),
        "cs": ("float", 0.01, 0.4, False),
        "error_decay": ("float", 0.98, 0.999, False),
    },
}


def fixed_params(opt, proto=PROTOCOL):
    """Parameters that are not searched. Lion is LPECO with correction_strength = 0."""
    if opt in ("adam", "lion"):
        return {"cs": 0.0, "error_decay": proto["unused_error_decay"]}
    return {}


def full_params(opt, searched, proto=PROTOCOL):
    p = dict(searched)
    p.update(fixed_params(opt, proto))
    return p


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def tuning_seeds(trial_number, n=3, max_seed=PROTOCOL["max_tuning_seed"]):
    """Trial k uses seeds 3k, 3k+1, 3k+2 (identical for every optimizer)."""
    seeds = [n * trial_number + j for j in range(n)]
    if max(seeds) > max_seed:
        raise ValueError(f"trial {trial_number} would use seed {max(seeds)} > {max_seed}")
    return seeds


# ---------------------------------------------------------------------------
# Learning-rate edge rule (outer 10% of the log range)
# ---------------------------------------------------------------------------
def lr_edges(opt):
    _, lo, hi, log = SPACES[opt]["lr"]
    assert log
    f = 10 ** (0.1 * math.log10(hi / lo))
    return lo * f, hi / f


def lr_at_edge(opt, lr):
    lo_edge, hi_edge = lr_edges(opt)
    return lr < lo_edge or lr > hi_edge


# ---------------------------------------------------------------------------
# Active time and the phase-2 decision (uses only run counts and timestamps, never returns)
# ---------------------------------------------------------------------------
def active_elapsed(start, finish_times, now, gap_s=PROTOCOL["gap_s"]):
    """Wall-clock time since `start` minus every gap longer than gap_s with no run completing."""
    total, prev = 0.0, start
    for t in sorted(finish_times):
        d = t - prev
        if d <= gap_s:
            total += d
        prev = t
    d = now - prev
    if 0 <= d <= gap_s:
        total += d
    return total


def long_gaps(start, finish_times, gap_s=PROTOCOL["gap_s"]):
    """(gap_start, gap_end) pairs longer than gap_s, for logging sleep/crash/restart periods."""
    out, prev = [], start
    for t in sorted(finish_times):
        if t - prev > gap_s:
            out.append((prev, t))
        prev = t
    return out


def make_decision(active_s, runs_completed, proto=PROTOCOL, now=None):
    return {
        "decided_at": now,
        "active_elapsed_s": active_s,
        "runs_completed": runs_completed,
        "min_runs": proto["min_runs_at_check"],
        "proceed_to_phase2": runs_completed >= proto["min_runs_at_check"],
    }


def target_trials(n_done, decision, proto=PROTOCOL):
    """How many completed trials an optimizer should reach. None = wait for the decision."""
    if n_done < proto["phase1_trials"]:
        return proto["phase1_trials"]
    if decision is None:
        return None
    return proto["phase2_trials"] if decision["proceed_to_phase2"] else proto["phase1_trials"]


# ---------------------------------------------------------------------------
# Jobs and files
# ---------------------------------------------------------------------------
def result_path(root, opt, trial_number, seed):
    return Path(root) / "tuning" / opt / f"trial{trial_number:03d}_seed{seed:04d}.json"


def build_job(opt, trial_number, seed_index, params, proto, root):
    seeds = tuning_seeds(trial_number, proto["seeds_per_trial"], proto["max_tuning_seed"])
    seed = seeds[seed_index]
    p = full_params(opt, params, proto)
    train_kwargs = dict(
        opt=opt,
        seed=seed,
        steps=proto["steps"],
        lr=float(p["lr"]),
        target_every=int(p["target_every"]),
        cs=float(p["cs"]),
        error_decay=float(p["error_decay"]),
        beta1=proto["beta1"],
        beta2=proto["beta2"],
        eval_every=proto["steps"],            # exactly one greedy evaluation, at the end
        eval_episodes=proto["eval_episodes"],
        learning_starts=proto["learning_starts"],
    )
    return {
        "out_path": str(result_path(root, opt, trial_number, seed)),
        "opt": opt,
        "trial": trial_number,
        "seed": seed,
        "train_kwargs": train_kwargs,
    }


def collect_finish_times(root):
    times = []
    for f in (Path(root) / "tuning").rglob("*.json"):
        try:
            times.append(json.loads(f.read_text())["finished_at"])
        except Exception:
            continue
    return times


def run_counts(trials_per_optimizer, n_opt=3, seeds_per_trial=3, recheck=75, final=90, exploratory=30):
    tuning = trials_per_optimizer * n_opt * seeds_per_trial
    return {"tuning": tuning, "total": tuning + recheck + final + exploratory}
