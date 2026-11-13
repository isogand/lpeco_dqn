"""Statistics for the pre-registered analysis (PROTOCOL.md Sections 2 and 6, DEVIATIONS.md D2).

Numpy + scipy only. Everything that decides an outcome is here, so it can be tested on synthetic data
before any final run exists.
"""
import numpy as np
from scipy import stats

FLOOR = 1.0            # denominator floor for the SD ratio inside resamples (Section 2, "Zero or tiny SD")
SATURATED = 490.0
COLLAPSED = 250.0
WEAK = 250.0
DELTA = 25.0
SENSITIVITY_DELTAS = (10.0, 25.0, 50.0)
N_CHECKPOINTS = 20     # 20 greedy checkpoints per final run; the late window is the second half


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------
def iqm(x):
    return float(stats.trim_mean(np.asarray(x, float), 0.25))


def checkpoint_means(evals):
    if len(evals) != N_CHECKPOINTS:
        raise ValueError(f"expected {N_CHECKPOINTS} checkpoints, got {len(evals)}")
    return np.array([r for _, r in evals], float)


def late_mean(evals):
    return float(checkpoint_means(evals)[N_CHECKPOINTS // 2:].mean())


def late_min(evals):
    return float(checkpoint_means(evals)[N_CHECKPOINTS // 2:].min())


def collapse_count(evals):
    """Checkpoints after the first one with mean >= 490 whose mean is < 250. None = never reached 490."""
    c = checkpoint_means(evals)
    first = np.flatnonzero(c >= SATURATED)
    if first.size == 0:
        return None
    return int(np.sum(c[first[0] + 1:] < COLLAPSED))


def run_metrics(runs):
    """runs: list of {"final_return", "evals"} -> dict of arrays (one entry per seed)."""
    final = np.array([r["final_return"] for r in runs], float)
    cc = [collapse_count(r["evals"]) for r in runs]
    return {
        "final": final,
        "late_mean": np.array([late_mean(r["evals"]) for r in runs]),
        "late_min": np.array([late_min(r["evals"]) for r in runs]),
        "collapses": cc,
    }


# ---------------------------------------------------------------------------
# Resampling statistics (two independent groups: x = LPECO or first group, l = Lion or second group)
# ---------------------------------------------------------------------------
def stat_sd_ratio(xb, lb, floor=FLOOR):
    return np.std(xb, axis=1, ddof=1) / np.maximum(np.std(lb, axis=1, ddof=1), floor)


def stat_iqm_diff(xb, lb):
    return stats.trim_mean(xb, 0.25, axis=1) - stats.trim_mean(lb, 0.25, axis=1)


def _observed(stat_fn, x, l):
    return float(stat_fn(x[None, :], l[None, :])[0])


def bca_bound(stat_fn, x, l, B, rng, side):
    """One-sided 95% BCa bound with independent resampling of the two groups.

    side = "upper": 95% upper bound; side = "lower": 95% lower bound.
    Returns (bound, observed). Acceleration comes from a jackknife over both groups' observations.
    """
    x, l = np.asarray(x, float), np.asarray(l, float)
    n, m = len(x), len(l)
    theta = _observed(stat_fn, x, l)
    boots = stat_fn(x[rng.integers(0, n, (B, n))], l[rng.integers(0, m, (B, m))])
    prop = (np.sum(boots < theta) + 0.5 * np.sum(boots == theta)) / B
    z0 = stats.norm.ppf(np.clip(prop, 1.0 / (B + 1), B / (B + 1.0)))
    jack = [_observed(stat_fn, np.delete(x, i), l) for i in range(n)]
    jack += [_observed(stat_fn, x, np.delete(l, j)) for j in range(m)]
    jack = np.array(jack)
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0
    zq = stats.norm.ppf(0.95) if side == "upper" else stats.norm.ppf(0.05)
    adj = stats.norm.cdf(z0 + (z0 + zq) / (1.0 - a * (z0 + zq)))
    return float(np.quantile(boots, float(np.clip(adj, 0.0, 1.0)))), theta


def perm_p_sd_ratio(x, l, B, rng):
    """One-sided permutation p-value for H1: SD(x) / max(SD(l), floor) is smaller than expected under exchangeability."""
    x, l = np.asarray(x, float), np.asarray(l, float)
    n = len(x)
    obs = _observed(stat_sd_ratio, x, l)
    pool = np.concatenate([x, l])
    shuffled = rng.permuted(np.tile(pool, (B, 1)), axis=1)
    perm = np.std(shuffled[:, :n], axis=1, ddof=1) / np.maximum(np.std(shuffled[:, n:], axis=1, ddof=1), FLOOR)
    return float((1 + np.sum(perm <= obs)) / (B + 1))


# ---------------------------------------------------------------------------
# Primary analysis: LPECO vs Lion, decision rule of Section 2
# ---------------------------------------------------------------------------
def primary_analysis(x_lpeco, l_lion, B=10_000, rng=None, deltas=SENSITIVITY_DELTAS):
    rng = rng or np.random.default_rng(12345)
    x, l = np.asarray(x_lpeco, float), np.asarray(l_lion, float)
    sd_x, sd_l = float(np.std(x, ddof=1)), float(np.std(l, ddof=1))
    out = {
        "n_lpeco": len(x), "n_lion": len(l),
        "sd_lpeco": sd_x, "sd_lion": sd_l,
        "sd_ratio_raw": (sd_x / sd_l) if sd_l > 0 else None,
        "sd_ratio_floored": sd_x / max(sd_l, FLOOR),
        "iqm_lpeco": iqm(x), "iqm_lion": iqm(l),
        "frac_ge490_lpeco": float(np.mean(x >= SATURATED)), "frac_ge490_lion": float(np.mean(l >= SATURATED)),
    }
    out["saturated"] = bool(out["frac_ge490_lpeco"] >= 0.9 and out["frac_ge490_lion"] >= 0.9)
    out["ratio_undefined"] = bool(sd_l == 0.0)

    ub, _ = bca_bound(stat_sd_ratio, x, l, B, rng, "upper")
    out["ratio_ci_upper95"] = ub
    out["perm_p"] = perm_p_sd_ratio(x, l, B, rng)
    lb, diff = bca_bound(stat_iqm_diff, x, l, B, rng, "lower")
    ub_d, _ = bca_bound(stat_iqm_diff, x, l, B, rng, "upper")
    out["iqm_diff"], out["iqm_diff_lower95"], out["iqm_diff_upper95"] = diff, lb, ub_d
    out["lpeco_worse"] = bool(ub_d < 0)
    out["brown_forsythe_p_two_sided"] = float(stats.levene(x, l, center="median").pvalue)

    out["cond1_ratio_upper_below_1"] = bool(ub < 1.0)
    out["cond2_perm_p_below_0.05"] = bool(out["perm_p"] < 0.05)
    out["cond3_by_delta"] = {str(d): bool(lb > -d) for d in deltas}

    def decide(delta):
        if out["saturated"]:
            return "non-discriminating"
        if out["ratio_undefined"]:
            return "no evidence for H1 (SD ratio undefined: SD(Lion) = 0)"
        ok = out["cond1_ratio_upper_below_1"] and out["cond2_perm_p_below_0.05"] and (lb > -delta)
        return "H1 supported" if ok else "no evidence for H1"

    out["decision"] = decide(DELTA)
    out["decision_by_delta"] = {str(d): decide(d) for d in deltas}
    return out


# ---------------------------------------------------------------------------
# Secondary / descriptive comparisons (no decision attached)
# ---------------------------------------------------------------------------
def sd_ratio_report(x, l, B=10_000, rng=None):
    rng = rng or np.random.default_rng(12345)
    x, l = np.asarray(x, float), np.asarray(l, float)
    ub, theta = bca_bound(stat_sd_ratio, x, l, B, rng, "upper")
    return {"sd_first": float(np.std(x, ddof=1)), "sd_second": float(np.std(l, ddof=1)),
            "ratio_floored": theta, "ratio_ci_upper95": ub, "perm_p": perm_p_sd_ratio(x, l, B, rng)}


def describe(values):
    v = np.asarray(values, float)
    q1, q3 = np.percentile(v, [25, 75])
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(np.std(v, ddof=1)), "iqm": iqm(v),
            "iqr": float(q3 - q1), "min": float(v.min()), "max": float(v.max()),
            "frac_below_250": float(np.mean(v < WEAK)), "frac_ge490": float(np.mean(v >= SATURATED))}


def collapse_summary(collapses):
    never = sum(c is None for c in collapses)
    counts = [c for c in collapses if c is not None]
    return {"never_reached_490": int(never),
            "mean_collapses_per_seed": float(np.mean(counts)) if counts else None,
            "frac_seeds_with_collapse": float(np.mean([c > 0 for c in counts])) if counts else None}


def run_analysis(runs_by_label, B=10_000, seed=12345):
    """runs_by_label: {"adam"|"lion"|"lpeco"|"lpeco_cs0": [ {"final_return", "evals"}, ... ]}"""
    rng = np.random.default_rng(seed)
    m = {k: run_metrics(v) for k, v in runs_by_label.items()}
    res = {"B": B, "bootstrap_seed": seed, "delta": DELTA}
    res["primary"] = primary_analysis(m["lpeco"]["final"], m["lion"]["final"], B=B, rng=rng)

    res["describe"] = {k: {"final": describe(v["final"]), "late_mean": describe(v["late_mean"]),
                           "late_min": describe(v["late_min"]), "collapses": collapse_summary(v["collapses"])}
                       for k, v in m.items()}
    res["secondary_lpeco_vs_lion"] = {
        "late_mean_sd_ratio": sd_ratio_report(m["lpeco"]["late_mean"], m["lion"]["late_mean"], B, rng),
        "late_min_sd_ratio": sd_ratio_report(m["lpeco"]["late_min"], m["lion"]["late_min"], B, rng),
    }
    res["descriptive_adam_references"] = {
        "adam_vs_lion_final_sd_ratio": sd_ratio_report(m["adam"]["final"], m["lion"]["final"], B, rng),
        "adam_vs_lpeco_final_sd_ratio": sd_ratio_report(m["adam"]["final"], m["lpeco"]["final"], B, rng),
    }
    if "lpeco_cs0" in m:
        res["exploratory_lpeco_vs_lpeco_cs0"] = sd_ratio_report(m["lpeco"]["final"], m["lpeco_cs0"]["final"], B, rng)
    return res


def format_report(res):
    p = res["primary"]
    lines = ["# Analysis report (pre-registered rule, PROTOCOL.md Section 2)", "",
             f"Bootstrap: BCa, B = {res['B']}, seed {res['bootstrap_seed']}, groups resampled independently. "
             f"Non-inferiority margin Delta = {res['delta']} (convention).", "",
             f"## Primary: LPECO vs Lion ({p['n_lpeco']} seeds each)", "",
             f"- DECISION: **{p['decision']}**",
             f"- SD of final return: LPECO {p['sd_lpeco']:.2f}, Lion {p['sd_lion']:.2f}; "
             f"ratio (floored) {p['sd_ratio_floored']:.3f}, raw {p['sd_ratio_raw'] if p['sd_ratio_raw'] is None else round(p['sd_ratio_raw'], 3)}",
             f"- Condition 1, upper 95% bound of the SD ratio < 1: {p['cond1_ratio_upper_below_1']} (bound {p['ratio_ci_upper95']:.3f})",
             f"- Condition 2, permutation p < 0.05: {p['cond2_perm_p_below_0.05']} (p = {p['perm_p']:.4f})",
             f"- Condition 3, IQM(LPECO) - IQM(Lion) = {p['iqm_diff']:.2f}, lower 95% bound {p['iqm_diff_lower95']:.2f} > -Delta: "
             f"{p['cond3_by_delta'][str(res['delta'])]}",
             f"- Saturation: share of seeds >= 490: LPECO {p['frac_ge490_lpeco']:.2f}, Lion {p['frac_ge490_lion']:.2f} "
             f"(rule: non-discriminating if both >= 0.90): {p['saturated']}",
             f"- LPECO worse than Lion (upper 95% bound of IQM difference < 0): {p['lpeco_worse']}",
             f"- Sensitivity of the decision to Delta: {p['decision_by_delta']}",
             f"- Brown-Forsythe (two-sided, supplementary): p = {p['brown_forsythe_p_two_sided']:.4f}", "",
             "'no evidence for H1' is absence of evidence, not equivalence (Section 2).", "",
             "## Secondary (not part of the decision rule)", ""]
    for name, key in (("late-window mean", "late_mean_sd_ratio"), ("late-window minimum", "late_min_sd_ratio")):
        r = res["secondary_lpeco_vs_lion"][key]
        lines.append(f"- {name}: SD LPECO {r['sd_first']:.2f} vs Lion {r['sd_second']:.2f}, ratio {r['ratio_floored']:.3f}, "
                     f"upper 95% bound {r['ratio_ci_upper95']:.3f}, permutation p {r['perm_p']:.4f}")
    lines += ["", "## Descriptive per group", ""]
    for k, d in res["describe"].items():
        f = d["final"]
        lines.append(f"- {k}: final mean {f['mean']:.1f}, SD {f['sd']:.1f}, IQM {f['iqm']:.1f}, IQR {f['iqr']:.1f}, "
                     f"share < 250: {f['frac_below_250']:.2f}, share >= 490: {f['frac_ge490']:.2f}; collapses: {d['collapses']}")
    lines += ["", "## Adam (descriptive reference only)", ""]
    for k, r in res["descriptive_adam_references"].items():
        lines.append(f"- {k}: ratio {r['ratio_floored']:.3f}, upper 95% bound {r['ratio_ci_upper95']:.3f}")
    if "exploratory_lpeco_vs_lpeco_cs0" in res:
        r = res["exploratory_lpeco_vs_lpeco_cs0"]
        lines += ["", "## Exploratory: LPECO vs the same configuration with cs = 0", "",
                  f"- SD LPECO {r['sd_first']:.2f} vs cs=0 {r['sd_second']:.2f}, ratio {r['ratio_floored']:.3f} "
                  "(the learning rate was tuned for LPECO, not for Lion: exploratory only)"]
    return "\n".join(lines) + "\n"
