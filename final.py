"""Final evaluation stage (PROTOCOL.md 5.4).

Runs the configuration selected by reeval.py for each optimizer with 30 fresh seeds (2000-2029), greedy evaluation every
25,000 steps (20 checkpoints x 20 episodes), plus the exploratory run: the selected LPECO configuration with
correction_strength = 0 (Lion). 120 runs, about 1.5 h with 4 workers.

    python3 final.py --smoke                                     # rehearsal; needs tune.py --smoke and reeval.py --smoke
    caffeinate -i python3 final.py --confirm-selection-committed

Reads  : <root>/selected_configs.json
Writes : <root>/final/<label>/seed<NNNN>.json   (label = adam, lion, lpeco, lpeco_cs0)
Resumable. This script prints no returns: use analyze.py once everything is finished.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import sys
from pathlib import Path

import dqn
import stages_lib as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--confirm-selection-committed", action="store_true",
                    help="required for the real run: selected_configs.json and analyze.py must already be committed")
    args = ap.parse_args()

    settings = S.stage_settings(args.smoke)
    root = args.root or ("smoke_experiment" if args.smoke else "experiment")
    if not args.smoke and not args.confirm_selection_committed:
        sys.exit("Refusing to start: commit selected_configs.json and the tested analyze.py first, "
                 "then pass --confirm-selection-committed.")
    sel_path = Path(root) / "selected_configs.json"
    if not sel_path.exists():
        sys.exit(f"{sel_path} not found: run reeval.py first.")
    selected = json.loads(sel_path.read_text())["selected"]

    jobs = S.build_final_jobs(root, selected, settings)
    pending = [j for j in jobs if not Path(j["out_path"]).exists()]
    print(f"final evaluation: {len(jobs)} runs ({len(settings['final_seeds'])} seeds x {len(S.ALL_FINAL_LABELS)} groups), "
          f"{len(jobs) - len(pending)} already done, {len(pending)} to run; {settings['steps']:,} steps each, "
          f"evaluation every {settings['final_eval_every']:,} steps")
    if not pending:
        print("nothing to do.")
        return
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
    print("All final runs finished. Next: commit the final/ folder, then python3 analyze.py")


if __name__ == "__main__":
    main()
