#!/usr/bin/env python3
"""Optuna hyper-parameter search for one (model, dataset) on one seed, selected on the validation set only.

    python benchmark/tune.py --model gcn --dataset abide_AAL116 --trials 500 --seed 42      # run / resume (many workers can share it)
    python benchmark/tune.py --model gcn --dataset abide_AAL116 --export                    # write best.json from the finished study

Protocol (identical for every model; see benchmark/spaces.py)
  * split = common.make_split(labels, seed): the same subjects as the final runs of that seed; test data is never read here
  * objective = lowest validation loss over the epochs of a trial (the same rule that picks checkpoints in run.py)
  * TPE sampler (multivariate, seeded) + median pruner on the per-epoch validation loss; a pruned trial still counts towards the budget
  * trial 0 = shared-recipe default, trial 1 (BioBGT) = the authors' recipe; both are evaluated before the sampler starts
  * the study lives in <out>/<model>/<dataset>/study.journal (Optuna journal file, safe for several workers on a shared disk)

Outputs in <out>/<model>/<dataset>/: study.journal, trials/<n>/train_log.jsonl, trials.csv and best.json (hp + train settings
that `run.py --tuned` consumes for the final seeds).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import spaces  # noqa: E402
from adapters import MODELS, load  # noqa: E402

BIG = 1e6  # reported for diverged trials so the sampler steers away from them


def progress_step(epoch, epochs):
    """Pruning step = percent of training done, so trials with 50 and 300 epochs (and their different lr schedules) are
    compared at the same point of their schedule rather than at the same epoch number."""
    return int(round(100 * (epoch + 1) / epochs))


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--dataset", required=True)
    p.add_argument("--trials", type=int, default=500, help="total budget of the study (all workers together)")
    p.add_argument("--seed", type=int, default=42, help="seed of the split and of the study")
    p.add_argument("--worker", type=int, default=0,
                   help="number of this worker within its study; the sampler seed is seed + 1000 * worker, so workers sharing a study "
                        "propose different configurations (identical seeds made them repeat each other)")
    p.add_argument("--data_dir", default=str(common.REPO / "Datasets"))
    p.add_argument("--out", default=str(common.REPO / "runs" / "_tune"))
    p.add_argument("--device", default="auto")
    p.add_argument("--cache", default=str(common.REPO / "runs" / "_cache"))
    p.add_argument("--max_epochs", type=int, default=None, help="cap epochs (smoke tests only; not for real studies)")
    p.add_argument("--gpu_fraction", type=float, default=None,
                   help="cap this process at this fraction of the GPU memory, so that several workers can share one GPU and an "
                        "out-of-memory trial only fails its own process")
    p.add_argument("--export", action="store_true", help="only write best.json / trials.csv from the existing study")
    p.add_argument("--tune_seeds", default=None,
                   help="comma list of split seeds whose validation sets are all used: a trial's value is the mean of its best "
                        "validation loss over these splits (each split trained from scratch with init seed = split seed). "
                        "Default: only --seed. Test partitions are never read. Pruning uses the first split only.")
    p.add_argument("--sampler_seed", type=int, default=None,
                   help="seed of the TPE sampler only (default: --seed). The split is still set by --seed.")
    p.add_argument("--enqueue_from", default=None,
                   help="trials.csv of an earlier study of the same model / dataset: its best finished trial is enqueued first")
    p.add_argument("--strict_determinism", action="store_true",
                   help="raise on any operation without a deterministic kernel (default: warn); use with a single worker "
                        "per study so the seeded TPE sampler proposes the same trial sequence on every rerun")
    p.add_argument("--select_by", choices=["val_loss", "val_auc"], default="val_loss",
                   help="metric used to pick the best trial during export (default: val_loss; use val_auc for "
                        "imbalanced multi-class datasets such as PPMI where low CE loss does not imply high AUC)")
    return p.parse_args()


def study_paths(a):
    d = Path(a.out) / a.model / a.dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_study(a, d):
    storage = JournalStorage(JournalFileBackend(str(d / "study.journal")))
    return optuna.create_study(
        study_name=f"{a.model}/{a.dataset}/seed{a.seed}", storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=(a.seed if getattr(a, "sampler_seed", None) is None else a.sampler_seed) + 1000 * getattr(a, "worker", 0), multivariate=True, n_startup_trials=30,
                                           constant_liar=True),  # trials still running count as bad, so parallel workers spread out
        pruner=optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=20))  # steps are percent of training


def oom_on_small_gpu(t):
    """True for a trial that ran out of memory on a GPU with less than 20 GB (or a MIG slice). Such failures depend on the
    hardware rather than the configuration, so these trials are not counted against the budget."""
    if "OutOfMemory" not in str(t.user_attrs.get("failure", "")):
        return False
    mem = t.user_attrs.get("gpu_mem_gb")
    return (mem is not None and mem < 20) or "MIG" in str(t.user_attrs.get("device", ""))


def select_budget(study, budget):
    """The first `budget` finished trials by trial number (workers that were already running when the budget was reached may
    add a few more; they are ignored, so every study is judged on exactly the same number of trials)."""
    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    finished = sorted((t for t in study.trials if t.state in states and not oom_on_small_gpu(t)), key=lambda t: t.number)[:budget]
    return finished, [t for t in finished if t.state == optuna.trial.TrialState.COMPLETE]


def _trial_best_auc(d, trial):
    """Return the max val_auc seen across epochs for a finished trial (from its train_log.jsonl).
    Returns -1 if the log is missing or unreadable."""
    log = d / "trials" / str(trial.number) / "train_log.jsonl"
    try:
        return max(json.loads(line).get("val_auc", -1) for line in open(log))
    except Exception:
        return -1


def export(a, d, study, info):
    finished, done = select_budget(study, a.trials)
    if not done:
        print("no completed trial yet")
        return
    select_by = getattr(a, "select_by", "val_loss")
    if select_by == "val_auc":
        scored = [(t, _trial_best_auc(d, t)) for t in done]
        scored = [(t, s) for t, s in scored if s >= 0]
        if not scored:
            print("WARNING: no val_auc logs found, falling back to val_loss selection")
            best = min(done, key=lambda t: (t.value, t.number))
            select_by = "val_loss"
        else:
            best, best_metric = max(scored, key=lambda x: (x[1], -x[0].number))
    else:
        best = min(done, key=lambda t: (t.value, t.number))
    hp, train = replay(best, a.model, info)
    train["selection"] = select_by  # run.py reads this to set epoch-selection criterion
    by_number = {t.number: t for t in study.trials}
    defaults = {n: by_number[n].value for n in (0, 1) if n in by_number and by_number[n].value is not None}
    metric_val = (_trial_best_auc(d, best) if select_by == "val_auc" else best.value)
    common.write_json(d / "best.json", {
        "model": a.model, "dataset": a.dataset, "tune_seed": a.seed, "trial": best.number,
        "val_loss": best.value, "select_by": select_by, f"best_{select_by}": metric_val,
        "budget": a.trials, "n_trials": len(finished), "n_complete": len(done),
        "n_pruned": len(finished) - len(done), "n_failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "n_ignored_extra": len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
                               and not oom_on_small_gpu(t)]) - len(finished),
        "n_oom_small_gpu_ignored": sum(oom_on_small_gpu(t) for t in study.trials), "default_val_loss": defaults.get(0), "second_enqueued_val_loss": defaults.get(1),
        "hp": hp, "train": train, "params": best.params})
    df = study.trials_dataframe()
    df.to_csv(d / "trials.csv", index=False)
    print(f"best trial {best.number} (selected by {select_by}={metric_val:.5f}, val_loss={best.value:.5f}) of {len(finished)} trials\n  train={train}\n  hp={hp}")


class _Replay:
    """Re-runs the suggest functions against the stored params of a finished trial."""

    def __init__(self, params):
        self.params = params

    def suggest_float(self, name, *args, **kwargs):
        return self.params[name]

    suggest_int = suggest_categorical = suggest_float


def replay(trial, model, info):
    r = _Replay(trial.params)
    return spaces.suggest_model(r, model, info), spaces.suggest_train(r, model)


def main():
    import torch
    a = parse()
    d = study_paths(a)
    bl = load(a.model)
    # Do NOT call torch.cuda.is_available() or any CUDA API before bl.inputs().
    # bl.inputs() (biobgt) uses multiprocessing.Pool (fork); CUDA init before fork
    # causes cudaErrorInitializationError in child workers.
    _want_cuda = (a.device == "auto") or (a.device == "cuda")
    data = common.load_dataset(common.find_dataset_file(a.dataset, a.data_dir), dedup=None,
                               with_timeseries=getattr(bl, "NEEDS_TIMESERIES", False))
    idx = common.make_split(data["labels"], a.seed, groups=data["groups"])
    tune_seeds = [int(x) for x in a.tune_seeds.split(",")] if getattr(a, "tune_seeds", None) else [a.seed]
    idxs = {s: common.make_split(data["labels"], s, groups=data["groups"]) for s in tune_seeds}
    common.write_json(d / "tune_seeds.json", {"tune_seeds": tune_seeds})
    common.write_json(d / "split_sha.json", {"split_sha256": common.split_record(data, a.seed, idx)["sha256"],
                                             "data_sha256": common.file_sha256(data["file"])})
    info = {"n_roi": int(data["corr"].shape[1]), "n_classes": data["n_classes"], "dataset_name": data["name"],
            "dataset_family": data["dataset"]}
    study = make_study(a, d)
    if a.export:
        export(a, d, study, info)
        return 0
    if hasattr(bl, "inputs"):
        data["inputs"] = bl.inputs(data, Path(a.cache) / a.model, {})
        info["n_time"] = int(data["inputs"][0].shape[-1])
    else:
        data["inputs"] = [data["corr"]]
    # Safe to initialize CUDA now — Pool workers have been forked and joined.
    _use_cuda = _want_cuda and (a.device == "cuda" or torch.cuda.is_available())
    device = torch.device("cuda" if _use_cuda else "cpu")
    if device.type == "cuda" and a.gpu_fraction:
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)

    if a.worker == 0 and not study.trials:  # only worker 0 enqueues the default / authors' configurations
        for params in spaces.default_trials(a.model, info):
            study.enqueue_trial(params, skip_if_exists=True)
        if a.enqueue_from:
            import pandas as pd
            prev = pd.read_csv(a.enqueue_from)
            prev = prev[prev["state"] == "COMPLETE"] if "state" in prev else prev
            row = prev.sort_values("value").iloc[0]
            params = {c[len("params_"):]: (v.item() if hasattr(v, "item") else v) for c, v in row.items()
                      if c.startswith("params_") and not pd.isna(v)}
            params = {k: int(v) if isinstance(v, float) and v.is_integer() and k not in ("lr", "weight_decay") else v
                      for k, v in params.items()}  # csv turns ints of optional params into floats (8 -> 8.0)
            print(f"enqueue best trial of {a.enqueue_from} (value {row['value']:.5f}): {params}")
            study.enqueue_trial(params, skip_if_exists=True)

    def objective(trial):
        train = spaces.suggest_train(trial, a.model)
        hp = spaces.suggest_model(trial, a.model, info)
        if a.max_epochs:
            train["epochs"] = min(train["epochs"], a.max_epochs)
        cfg = {"epochs": train["epochs"], "batch_size": train["batch_size"], "lr_final_ratio": train["lr_final_ratio"],
               "grad_clip": train["grad_clip"], "mixup": train["mixup"], "optimizer": train["optimizer"],
               "schedule": train["schedule"], "warmup_epochs": train["warmup_epochs"], "selection": "val_loss",
               "init_seed": a.seed, "skip_test": True}
        tdir = d / "trials" / str(trial.number)
        tdir.mkdir(parents=True, exist_ok=True)

        last = {"step": -1}

        def on_epoch(epoch, val_loss):
            step = progress_step(epoch, cfg["epochs"])
            if step == last["step"]:  # several epochs can share one percent step; Optuna accepts one report per step
                return
            last["step"] = step
            trial.report(val_loss, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        losses, outs = [], []
        for k, s in enumerate(tune_seeds):
            common.set_seed(s, strict=getattr(a, "strict_determinism", False))
            model, forward, groups = bl.build(info, hp, train["lr"], train["weight_decay"])
            model = model.to(device)
            if k == 0:
                common.write_json(tdir / "params.json", {"hp": hp, "train": train, "tune_seeds": tune_seeds,
                                                         "n_params": sum(p.numel() for p in model.parameters())})
            sdir = tdir if len(tune_seeds) == 1 else tdir / f"split_{s}"
            sdir.mkdir(parents=True, exist_ok=True)
            try:
                o = common.fit(model, forward, groups, data, idxs[s], sdir, {**cfg, "init_seed": s}, device, log=lambda m: None,
                               on_epoch=on_epoch if k == 0 else None)
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as e:
                trial.set_user_attr("failure", repr(e))
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                return BIG
            losses.append(o["best_val_loss"]); outs.append(o)
        out = {"best_epoch": outs[0]["best_epoch"], "train_seconds": sum(o["train_seconds"] for o in outs),
               "best_val_loss": float(sum(losses) / len(losses))}
        if len(tune_seeds) > 1:
            trial.set_user_attr("val_loss_per_split", {str(s): float(v) for s, v in zip(tune_seeds, losses)})
        trial.set_user_attr("worker_pid", os.getpid())
        trial.set_user_attr("worker", a.worker)
        trial.set_user_attr("device", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
        if device.type == "cuda":
            trial.set_user_attr("gpu_mem_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
        trial.set_user_attr("best_epoch", out["best_epoch"])
        trial.set_user_attr("seconds", round(out["train_seconds"], 1))
        return out["best_val_loss"]

    FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    n_done = sum(t.state in FINISHED for t in study.trials)  # crashed or preempted trials do not use up the budget
    if n_done >= a.trials:
        print(f"budget already used: {n_done} trials")
        export(a, d, study, info)
        return 0
    stop = optuna.study.MaxTrialsCallback(a.trials, states=FINISHED)
    t0 = time.time()
    study.optimize(objective, callbacks=[stop], gc_after_trial=True)
    print(f"{a.model}/{a.dataset}: {len(study.trials)} trials in {time.time() - t0:.0f}s on this worker")
    export(a, d, study, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
