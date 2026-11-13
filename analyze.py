"""Pre-registered analysis (PROTOCOL.md Sections 2 and 6, DEVIATIONS.md D2).

    python3 analyze.py --synthetic        # self-test on synthetic data with known truth (must pass BEFORE the final runs)
    python3 analyze.py                    # real analysis of <root>/final   (root default: experiment)
    python3 analyze.py --smoke            # rehearsal on ./smoke_experiment
    python3 analyze.py --check-only       # validate the structure of <root>/final; prints NO returns or statistics

The bootstrap/permutation seed is fixed (12345). Output: <root>/analysis/report.md, results.json and, if matplotlib
is installed, curves.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import analysis_lib as A

LABELS = ("adam", "lion", "lpeco", "lpeco_cs0")


def load_final_runs(root, expected):
    runs, seeds = {}, {}
    for label in LABELS:
        files = sorted((Path(root) / "final" / label).glob("seed*.json"))
        runs[label] = [json.loads(f.read_text()) for f in files]
        seeds[label] = sorted(r["seed"] for r in runs[label])
        if len(files) != expected:
            sys.exit(f"{label}: {len(files)} result files in {root}/final/{label}, expected {expected}. "
                     "Finish final.py first (it is resumable).")
    if not (seeds["adam"] == seeds["lion"] == seeds["lpeco"] == seeds["lpeco_cs0"]):
        sys.exit("the four groups do not use the same seed set")
    return runs


def check_structure(root, runs, expected, final_steps):
    """Structural validation only: counts, seeds, 20 checkpoints, and that every run used the selected configuration."""
    sel = json.loads((Path(root) / "selected_configs.json").read_text())["selected"]
    problems = []
    for label, rs in runs.items():
        cfg_key = "lpeco" if label == "lpeco_cs0" else label
        cfg = sel[cfg_key]["config"]
        for r in rs:
            if len(r["evals"]) != A.N_CHECKPOINTS:
                problems.append(f"{label} seed {r['seed']}: {len(r['evals'])} checkpoints")
            elif r["evals"][-1][0] != final_steps or r["final_return"] != r["evals"][-1][1]:
                problems.append(f"{label} seed {r['seed']}: final return is not the last checkpoint at step {final_steps}")
            c = r["config"]
            want_opt = "lion" if label == "lpeco_cs0" else label
            want_cs = 0.0 if label in ("lpeco_cs0", "adam", "lion") else cfg["cs"]
            if (r["opt"] != want_opt or c["lr"] != cfg["lr"] or c["target_every"] != cfg["target_every"]
                    or c["cs"] != want_cs or c["error_decay"] != cfg["error_decay"]):
                problems.append(f"{label} seed {r['seed']}: configuration differs from selected_configs.json")
            if not (2000 <= r["seed"] <= 2999):
                problems.append(f"{label}: seed {r['seed']} outside the final range")
    if problems:
        print("PROBLEMS:")
        for x in problems[:20]:
            print("  -", x)
        return False
    print(f"structure OK: {len(runs)} groups x {expected} seeds, {A.N_CHECKPOINTS} checkpoints each, "
          f"all runs use the configurations in selected_configs.json, same seed set in every group.")
    return True


# ---------------------------------------------------------------------------
# Synthetic self-test (known truth)
# ---------------------------------------------------------------------------
def _runs_from_finals(finals, rng, late_noise):
    out = []
    for f in finals:
        base = np.linspace(100.0, f, A.N_CHECKPOINTS)
        base[A.N_CHECKPOINTS // 2:] += rng.normal(0, late_noise, A.N_CHECKPOINTS // 2)
        base[-1] = f
        base = np.clip(base, 0.0, 500.0)
        out.append({"final_return": float(f), "evals": [[25_000 * (i + 1), float(v)] for i, v in enumerate(base)]})
    return out


SCENARIOS = {
    # name: (lpeco finals, lion finals, expected decision at Delta = 25)
    "SD cut by about 4x, mean not worse": (lambda r: np.clip(r.normal(420, 20, 30), 0, 500),
                                          lambda r: np.clip(r.normal(400, 90, 30), 0, 500), "H1 supported"),
    "no difference": (lambda r: np.clip(r.normal(380, 60, 30), 0, 500),
                      lambda r: np.clip(r.normal(380, 60, 30), 0, 500), "no evidence for H1"),
    "both at the ceiling": (lambda r: np.array([500.0] * 29 + [320.0]),
                            lambda r: np.array([500.0] * 28 + [210.0, 120.0]), "non-discriminating"),
    "Lion has zero SD, LPECO worse": (lambda r: np.array([500.0] * 12 + [100.0] * 18),
                                      lambda r: np.full(30, 500.0),
                                      "no evidence for H1 (SD ratio undefined: SD(Lion) = 0)"),
}


def synthetic_selftest(B):
    ok = True
    for i, (name, (mk_x, mk_l, expected)) in enumerate(SCENARIOS.items()):
        rng = np.random.default_rng(100 + i)
        x, l = mk_x(rng), mk_l(rng)
        runs = {"adam": _runs_from_finals(mk_l(rng), rng, 20), "lion": _runs_from_finals(l, rng, 20),
                "lpeco": _runs_from_finals(x, rng, 20), "lpeco_cs0": _runs_from_finals(mk_l(rng), rng, 20)}
        res = A.run_analysis(runs, B=B, seed=12345)
        got = res["primary"]["decision"]
        status = "ok" if got == expected else "MISMATCH"
        ok &= got == expected
        print(f"[{status}] {name}: decision = {got!r} (expected {expected!r})")
    # the report formatter must work on the last scenario too
    A.format_report(res)
    print("synthetic self-test:", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
def plot_curves(runs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available: no figure written.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for label, color in (("adam", "tab:gray"), ("lion", "tab:blue"), ("lpeco", "tab:red"), ("lpeco_cs0", "tab:green")):
        C = np.array([A.checkpoint_means(r["evals"]) for r in runs[label]])
        steps = [s for s, _ in runs[label][0]["evals"]]
        ax[0].plot(steps, C.mean(0), color=color, label=label)
        ax[0].fill_between(steps, C.mean(0) - C.std(0, ddof=1), C.mean(0) + C.std(0, ddof=1), color=color, alpha=0.12)
    ax[0].set_xlabel("environment steps"); ax[0].set_ylabel("greedy return"); ax[0].set_title("mean ± SD across seeds")
    ax[0].legend()
    for k, label in enumerate(("adam", "lion", "lpeco", "lpeco_cs0")):
        f = [r["final_return"] for r in runs[label]]
        ax[1].scatter(np.full(len(f), k) + np.random.default_rng(0).uniform(-0.15, 0.15, len(f)), f, s=14)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["adam", "lion", "lpeco", "lpeco_cs0"])
    ax[1].set_ylabel("final return"); ax[1].set_title("final return per seed")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"figure: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="self-test on synthetic data and exit")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--B", type=int, default=None, help="bootstrap/permutation resamples (protocol: 10,000)")
    ap.add_argument("--check-only", action="store_true", help="validate the structure of the final runs and exit")
    args = ap.parse_args()

    if args.synthetic:
        sys.exit(0 if synthetic_selftest(args.B or 2000) else 1)

    root = args.root or ("smoke_experiment" if args.smoke else "experiment")
    expected = 6 if args.smoke else 30
    B = args.B or (500 if args.smoke else 10_000)
    runs = load_final_runs(root, expected)
    if args.check_only:
        final_steps = 3_000 if args.smoke else 500_000
        sys.exit(0 if check_structure(root, runs, expected, final_steps) else 1)
    res = A.run_analysis(runs, B=B, seed=12345)
    text = A.format_report(res)
    out = Path(root) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(text)
    print(f"written: {out}/report.md and results.json")
    plot_curves(runs, out / "curves.png")


if __name__ == "__main__":
    main()
