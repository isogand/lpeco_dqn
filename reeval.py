"""Re-evaluation stage (PROTOCOL.md 5.3, DEVIATIONS.md D1).

For each optimizer: take the 5 best tuning trials (exact ties -> lower trial number), run each with 5 fresh seeds
(1000-1024, same seed sets for all optimizers), then select the configuration with the highest mean final return
(exact ties -> lower trial number; variance plays no role). 75 runs, about 1.5 h with 4 workers.

    python3 reeval.py --smoke                                   # rehearsal; needs: python3 tune.py --smoke
    caffeinate -i python3 reeval.py --confirm-deviations-committed

Reads  : <root>/tuning/*/*.json            (written by tune.py)
Writes : <root>/reeval/<opt>/*.json and <root>/selected_configs.json (never overwritten with different content)
Resumable: finished runs are not recomputed.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import sys
from pathlib import Path

import dqn
import stages_lib as S
import tune_lib as T


def fmt_cfg(c):
    s = f"lr={c['lr']:.3g} target_every={int(c['target_every'])}"
    if c["cs"] > 0:
        s += f" cs={c['cs']:.3f} error_decay={c['error_decay']:.4f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny rehearsal on ./smoke_experiment")
    ap.add_argument("--root", default=None)
    ap.add_argument("--confirm-deviations-committed", action="store_true",
                    help="required for the real run: DEVIATIONS.md (D1, D2) must be committed first")
    args = ap.parse_args()

    settings = S.stage_settings(args.smoke)
    root = args.root or ("smoke_experiment" if args.smoke else "experiment")
    if not args.smoke and not args.confirm_deviations_committed:
        sys.exit("Refusing to start: commit DEVIATIONS.md (D1, D2) first, then pass --confirm-deviations-committed.")

    # 1) the top-5 per optimizer, from the tuning files only
    tops = {}
    for opt in T.OPTIMIZERS:
        trials = S.load_tuning_trials(root, opt)
        if len(trials) < S.TOPK:
            sys.exit(f"{opt}: only {len(trials)} tuning trials found in {root}/tuning")
        tops[opt] = S.top_k(trials)
        print(f"\n{opt}: {len(trials)} tuning trials; top-{S.TOPK} (ties -> lower trial number)")
        for e in tops[opt]:
            print(f"  rank {e['rank']}: trial {e['trial']:>3}  tuning value {e['value']:7.2f}  {fmt_cfg(e['config'])}")

    # 2) run the 75 re-evaluation jobs
    jobs = S.build_reeval_jobs(root, tops, settings)
    pending = [j for j in jobs if not Path(j["out_path"]).exists()]
    print(f"\nre-evaluation: {len(jobs)} runs, {len(jobs) - len(pending)} already done, {len(pending)} to run "
          f"({settings['steps']:,} steps each, {settings['workers']} workers)")
    if pending:
        pool = cf.ProcessPoolExecutor(max_workers=settings["workers"], mp_context=mp.get_context("spawn"))
        try:
            futures = [pool.submit(dqn.run_job, j) for j in pending]
            for i, f in enumerate(cf.as_completed(futures), 1):
                f.result()
                if i % 5 == 0 or i == len(futures):
                    print(f"  {i}/{len(futures)} finished", flush=True)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print("Interrupted. Run the same command again to resume; finished runs are kept.")
            sys.exit(130)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # 3) selection (D1: highest mean, exact ties -> lower trial number)
    payload = {"rule": "highest mean final return over 5 fresh seeds; exact ties -> lower tuning trial number (D1)",
               "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()},
               "selected": {}, "reeval_table": {}}
    print("\nRE-EVALUATION RESULTS (mean final return over 5 seeds)")
    for opt in T.OPTIMIZERS:
        rows = S.load_reeval_rows(root, opt)
        if len(rows) != S.TOPK:
            sys.exit(f"{opt}: expected {S.TOPK} re-evaluated configurations, found {len(rows)}")
        chosen = S.select_after_reeval(rows)
        cfg = chosen["config"]
        payload["reeval_table"][opt] = [{"rank": r["rank"], "trial": r["trial"], "mean": r["mean"], "values": r["values"]}
                                        for r in rows]
        payload["selected"][opt] = {
            "trial": chosen["trial"], "rank": chosen["rank"], "reeval_mean": chosen["mean"],
            "reeval_values": chosen["values"], "config": cfg, "lr_at_edge": bool(T.lr_at_edge(opt, cfg["lr"])),
        }
        print(f"\n{opt}:")
        for r in rows:
            mark = "  <-- selected" if r["rank"] == chosen["rank"] else ""
            print(f"  rank {r['rank']} trial {r['trial']:>3}: mean {r['mean']:7.2f}  {[round(v, 1) for v in r['values']]}{mark}")
        print(f"  selected: trial {chosen['trial']}  {fmt_cfg(cfg)}  lr at edge of range: {payload['selected'][opt]['lr_at_edge']}")

    out = Path(root) / "selected_configs.json"
    if out.exists():
        if json.loads(out.read_text()) != json.loads(json.dumps(payload)):
            sys.exit(f"{out} already exists with DIFFERENT content. Refusing to overwrite a recorded selection.")
        print(f"\n{out} already recorded (identical).")
    else:
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwritten: {out}")
    print("Next: commit the reeval/ folder and selected_configs.json, then run final.py.")


if __name__ == "__main__":
    main()
