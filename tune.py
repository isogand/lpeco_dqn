"""Tuning stage of PROTOCOL.md (section 5.2): 3 Optuna studies (adam / lion / lpeco), one SQLite file each,
sharing a pool of 4 worker processes.

    python3 tune.py --dry-run                         # print the plan, run nothing
    python3 tune.py --smoke                           # tiny end-to-end rehearsal in ./smoke_experiment
    caffeinate -i python3 tune.py --confirm-protocol-committed    # the real thing (resumable)

Resuming: just run the same command again. Finished runs are never recomputed (one JSON file per run),
and trials left RUNNING by an interrupted launch are finished first.

The phase-2 decision (trials 49-64) uses ONLY run counts and timestamps, never any returns.
Written for Optuna 5.0.0 using only the basic API (create_study, ask, tell, suggest_*), but I could not run it
against Optuna here: the --smoke run on your machine is the real test.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import platform
import sys
import threading
import time
from pathlib import Path
from statistics import mean

import optuna

import dqn
import tune_lib as T

EVENT_LOCK = threading.Lock()
TS = optuna.trial.TrialState


def log_event(root, kind, **kw):
    rec = {"t": time.time(), "kind": kind, **kw}
    p = Path(root) / "logs" / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOCK:
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(time.strftime("%H:%M:%S"), kind, kw, flush=True)


def make_study(opt, root, proto):
    db = Path(root).resolve() / "studies" / f"{opt}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=proto["sampler_seed"], n_startup_trials=proto["n_startup_trials"])
    return optuna.create_study(
        study_name=opt,
        storage=f"sqlite:///{db}",
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )


def suggest(trial, opt):
    params = {}
    for name, spec in T.SPACES[opt].items():
        if spec[0] == "float":
            _, lo, hi, log = spec
            params[name] = trial.suggest_float(name, lo, hi, log=log)
        else:
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def evaluate_trial(pool, opt, number, params, proto, root):
    jobs = [T.build_job(opt, number, j, params, proto, root) for j in range(proto["seeds_per_trial"])]
    results = [f.result() for f in [pool.submit(dqn.run_job, job) for job in jobs]]
    finals = [r["final_return"] for r in results]
    if any(x is None for x in finals):
        raise RuntimeError(f"{opt} trial {number}: a run returned no final evaluation")
    return mean(finals), finals


def driver(opt, root, proto, pool, decision, decided, stop, errors):
    try:
        study = make_study(opt, root, proto)  # created inside this thread (one thread per SQLite file)

        # 1) finish trials left RUNNING by an interrupted launch
        running = sorted(study.get_trials(deepcopy=False, states=(TS.RUNNING,)), key=lambda t: t.number)
        for t in running:
            if stop.is_set():
                return
            if not set(T.SPACES[opt]).issubset(t.params):
                study.tell(t.number, state=TS.FAIL)  # asked but never received its parameters
                log_event(root, "trial_failed_incomplete_params", opt=opt, trial=t.number)
                continue
            value, finals = evaluate_trial(pool, opt, t.number, dict(t.params), proto, root)
            study.tell(t.number, value)
            log_event(root, "trial_done_resumed", opt=opt, trial=t.number, value=value, finals=finals)

        # 2) main loop
        while not stop.is_set():
            n_done = len(study.get_trials(deepcopy=False, states=(TS.COMPLETE,)))
            target = T.target_trials(n_done, decision["d"], proto)
            if target is None:  # phase 1 finished; wait for the (count/time-only) decision
                decided.wait(5)
                continue
            if n_done >= target:
                break
            trial = study.ask()
            params = suggest(trial, opt)
            value, finals = evaluate_trial(pool, opt, trial.number, params, proto, root)
            study.tell(trial, value)
            log_event(root, "trial_done", opt=opt, trial=trial.number, value=round(value, 2),
                      finals=finals, params={k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()})
        log_event(root, "driver_finished", opt=opt, completed=len(study.get_trials(deepcopy=False, states=(TS.COMPLETE,))))
    except BaseException as e:  # noqa: BLE001
        errors.append((opt, repr(e)))
        log_event(root, "driver_error", opt=opt, error=repr(e))
        stop.set()


def monitor(root, proto, decision, decided, stop, poll_s):
    path = Path(root) / "phase_decision.json"
    if path.exists():
        decision["d"] = json.loads(path.read_text())
        decided.set()
        return
    start = json.loads((Path(root) / "tuning_start.json").read_text())["started_at"]
    logged = set()
    while not stop.is_set():
        finishes = T.collect_finish_times(root)
        for g in T.long_gaps(start, finishes, proto["gap_s"]):
            if g not in logged:
                logged.add(g)
                log_event(root, "long_gap_excluded", start=g[0], end=g[1], seconds=round(g[1] - g[0]))
        t = time.time()
        act = T.active_elapsed(start, finishes, t, proto["gap_s"])
        if act >= proto["check_after_active_s"]:
            d = T.make_decision(act, len(finishes), proto, now=t)
            path.write_text(json.dumps(d))
            decision["d"] = d
            decided.set()
            log_event(root, "phase_decision", **d)
            return
        stop.wait(poll_s)


def init_root(root, proto):
    root = Path(root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    f = root / "tuning_start.json"
    if f.exists():
        info = json.loads(f.read_text())
        if info["protocol"] != json.loads(json.dumps(proto)):
            sys.exit(f"{f} was created with different protocol constants. Use a new --root, "
                     "or document the change in DEVIATIONS.md first.")
        log_event(root, "restart_or_resume")
        return info
    import gymnasium
    import numpy
    import torch
    info = {
        "started_at": time.time(),
        "protocol": proto,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"torch": torch.__version__, "numpy": numpy.__version__,
                     "gymnasium": gymnasium.__version__, "optuna": optuna.__version__},
    }
    f.write_text(json.dumps(info, indent=2))
    log_event(root, "tuning_started", versions=info["versions"])
    return info


def print_plan(proto, root):
    print(f"root folder           : {root}")
    print(f"optimizers            : {', '.join(T.OPTIMIZERS)}  (study per optimizer, TPE seed {proto['sampler_seed']}, "
          f"n_startup_trials {proto['n_startup_trials']})")
    print(f"steps per run         : {proto['steps']:,}   eval episodes at the end: {proto['eval_episodes']}")
    print(f"seeds per trial       : {proto['seeds_per_trial']}  (trial k uses seeds 3k..3k+2)")
    print(f"trials                : phase 1 = {proto['phase1_trials']}, phase 2 up to {proto['phase2_trials']}")
    print(f"phase-2 rule          : after {proto['check_after_active_s']}s of ACTIVE time, need >= "
          f"{proto['min_runs_at_check']} completed runs (gaps > {proto['gap_s']}s are not counted)")
    for k in (proto["phase1_trials"], proto["phase2_trials"]):
        c = T.run_counts(k, seeds_per_trial=proto["seeds_per_trial"])
        hours = c["total"] * 500_000 / 11_356 / 3600
        print(f"  {k} trials: {c['tuning']} tuning runs ({c['total']} runs in the whole experiment, "
              f"~{hours:.1f} h at the measured 4-worker rate)")
    for opt in T.OPTIMIZERS:
        print(f"search space {opt:5s}: {T.SPACES[opt]}")
        if opt != "lpeco":
            print(f"   fixed: {T.fixed_params(opt, proto)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny rehearsal (3000 steps/run); never for real results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--confirm-protocol-committed", action="store_true",
                    help="required for the real run: PROTOCOL.md must already be committed and tagged")
    args = ap.parse_args()

    proto = T.SMOKE if args.smoke else T.PROTOCOL
    root = args.root or ("smoke_experiment" if args.smoke else "experiment")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print_plan(proto, root)
    if args.dry_run:
        return
    if not args.smoke and not args.confirm_protocol_committed:
        sys.exit("Refusing to start the real tuning: commit and tag PROTOCOL.md first, then pass "
                 "--confirm-protocol-committed.")

    init_root(root, proto)
    decision, decided, stop, errors = {"d": None}, threading.Event(), threading.Event(), []
    poll_s = 1 if args.smoke else 30
    mon = threading.Thread(target=monitor, args=(root, proto, decision, decided, stop, poll_s), daemon=True)
    mon.start()

    ctx = mp.get_context("spawn")
    pool = cf.ProcessPoolExecutor(max_workers=proto["workers"], mp_context=ctx)
    threads = [threading.Thread(target=driver, args=(opt, root, proto, pool, decision, decided, stop, errors))
               for opt in T.OPTIMIZERS]
    try:
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    except KeyboardInterrupt:
        stop.set()
        log_event(root, "interrupted_by_user")
        print("Interrupted. Run the same command again to resume.")
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)

    if errors:
        print("ERRORS:", errors)
        sys.exit(1)

    summary = {}
    for opt in T.OPTIMIZERS:
        study = make_study(opt, root, proto)
        done = study.get_trials(deepcopy=False, states=(TS.COMPLETE,))
        best = study.best_trial
        summary[opt] = {"completed_trials": len(done), "best_value": best.value, "best_params": best.params,
                        "best_lr_at_edge": T.lr_at_edge(opt, best.params["lr"])}
    (Path(root) / "tuning_done.json").write_text(json.dumps({"summary": summary, "decision": decision["d"]}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
