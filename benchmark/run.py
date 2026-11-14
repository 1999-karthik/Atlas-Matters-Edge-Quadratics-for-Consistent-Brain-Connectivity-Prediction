#!/usr/bin/env python3
"""Run one (dataset, model, seed) experiment and write all artifacts to <out>/<model>/<dataset family>/<atlas>/seed_<seed>/.

    python benchmark/run.py --dataset abide_AAL116 --model gcn --seed 42 --data_dir Datasets --out runs

Layout: <out>/<model>/<dataset family>/<atlas>/seed_<seed>/, e.g. runs/gcn/abide/AAL116/seed_42/
Files written: run.log (full stdout/stderr), reproduce.json (exact command, data file hash, versions), config.json,
split.json, train_log.jsonl, best.pt, val_predictions.csv, test_predictions.csv (test rows only), metrics.json (or FAILED.json
with the traceback). A finished run is reused only if its settings fingerprint matches; otherwise it is rerun.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from adapters import MODELS, load  # noqa: E402


class Tee:
    """Duplicate a stream into the run's log file."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, x):
        self.stream.write(x); self.f.write(x); self.f.flush()

    def flush(self):
        self.stream.flush(); self.f.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="file stem, e.g. abide_AAL116, ppmi_ward100, adhd")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--seed", type=int, required=True, help="run seed; default for both split_seed and init_seed")
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--init_seed", type=int, default=None)
    p.add_argument("--data_dir", default=str(common.REPO / "Datasets"))
    p.add_argument("--out", default=str(common.REPO / "runs"))
    p.add_argument("--epochs", type=int, default=None, help="default 200 (or the value in --tuned)")
    p.add_argument("--tuned", default=None, help="best.json written by tune.py: its hp and train settings become the defaults "
                   "(explicit command-line options still win)")
    p.add_argument("--batch_size", type=int, default=None, help="default 16 (a baseline's build.py may override)")
    p.add_argument("--lr", type=float, default=None, help="default 1e-4 (a baseline's build.py may override)")
    p.add_argument("--lr_final_ratio", type=float, default=None, help="cosine decays lr to lr*ratio (default 0.1)")
    p.add_argument("--weight_decay", type=float, default=None, help="default 1e-4 (a baseline's build.py may override)")
    p.add_argument("--grad_clip", type=float, default=None, help="0 = off (default) (BNT, ALTER, BQN, EDGEQUADD train without clipping)")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    p.add_argument("--schedule", choices=["cosine", "warmup_cosine"], default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--mixup", choices=["hard", "soft", "none"], default=None,
                   help="default: soft = standard mixup, CE against the mixed one-hot label (what BNT/ALTER/BQN do); "
                        "hard = CE on argmax of the mixed label (EDGEQUADD's original code); per-model override in build.py")
    p.add_argument("--selection", choices=["val_loss", "val_auc"], default="val_loss")
    p.add_argument("--keep_duplicates", action="store_true", help="never drop exact duplicate subjects (default: drop, except ADHD-200, which is kept as released)")
    p.add_argument("--hp", type=json.loads, default={}, help='model overrides as JSON, e.g. \'{"dropout": 0.3}\'')
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--strict_determinism", action="store_true",
                   help="raise on any operation without a deterministic kernel (default: warn)")
    p.add_argument("--force", action="store_true", help="rerun even if metrics.json exists")
    return p.parse_args(argv)


def resolve(a):
    """Fill every unset option from the tuned file, the model's own overrides and the shared defaults.
    Returns (bl, settings): `settings` is what determines the run's result (also hashed into the fingerprint)."""
    bl = load(a.model)
    ov = dict(getattr(bl, "TRAIN_OVERRIDES", {}))
    tuned = json.load(open(a.tuned)) if a.tuned else {}
    ov.update(tuned.get("train", {}))  # tuned settings replace a model's own overrides
    a.hp = {**tuned.get("hp", {}), **a.hp}
    pick = lambda cli, key, default: cli if cli is not None else ov.get(key, default)  # noqa: E731
    a.epochs = pick(a.epochs, "epochs", 200)
    a.grad_clip = pick(a.grad_clip, "grad_clip", 0.0)
    a.lr, a.weight_decay = pick(a.lr, "lr", 1e-4), pick(a.weight_decay, "weight_decay", 1e-4)
    a.batch_size = pick(a.batch_size, "batch_size", 16)
    a.mixup = pick(a.mixup, "mixup", "soft")
    a.optimizer, a.schedule = pick(a.optimizer, "optimizer", "adam"), pick(a.schedule, "schedule", "cosine")
    a.warmup_epochs = pick(a.warmup_epochs, "warmup_epochs", 0)
    a.lr_final_ratio = pick(a.lr_final_ratio, "lr_final_ratio", 0.1)
    split_seed = a.seed if a.split_seed is None else a.split_seed
    init_seed = a.seed if a.init_seed is None else a.init_seed
    data_file = common.find_dataset_file(a.dataset, a.data_dir)
    settings = {
        "dataset": a.dataset, "model": a.model, "split_seed": split_seed, "init_seed": init_seed,
        "split_version": common.SPLIT_VERSION, "data_sha256": common.file_sha256(data_file), "keep_duplicates": bool(a.keep_duplicates),
        "train": {"epochs": a.epochs, "batch_size": a.batch_size, "lr_final_ratio": a.lr_final_ratio,
                  "grad_clip": a.grad_clip, "mixup": a.mixup, "optimizer": a.optimizer, "schedule": a.schedule,
                  "warmup_epochs": a.warmup_epochs, "selection": a.selection, "init_seed": init_seed},
        "lr": a.lr, "weight_decay": a.weight_decay, "hp": {**bl.DEFAULT_HP, **a.hp},
        "tuned_sha256": common.file_sha256(a.tuned) if a.tuned else None, "tuned_seed": tuned.get("tune_seed")}
    return bl, data_file, settings


def main():
    a = parse()
    import torch
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    bl, data_file, settings = resolve(a)
    split_seed, init_seed = settings["split_seed"], settings["init_seed"]
    fingerprint = common.settings_fingerprint(settings)

    rd = common.RunDir(a.out, a.dataset, a.model, a.seed)
    if rd.done(fingerprint) and not a.force:
        print(f"skip (done, same settings): {rd.path}")
        return 0
    if rd.done():
        print(f"settings changed since the finished run in {rd.path}: rerunning")
    run_dir = rd.fresh()
    logf = open(run_dir / "run.log", "w")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, logf), Tee(sys.__stderr__, logf)
    try:
        t0 = time.time()
        data = common.load_dataset(data_file, dedup=False if a.keep_duplicates else None,
                                   with_timeseries=getattr(bl, "NEEDS_TIMESERIES", False))
        idx = common.make_split(data["labels"], split_seed, groups=data["groups"])
        split = common.split_record(data, split_seed, idx)
        common.write_json(run_dir / "split.json", {**split, "dataset": data["name"], "file": data["file"],
                                                   "duplicates": data["dup_info"], "n_classes": data["n_classes"]})
        import dataset_info
        warns = dataset_info.data_warnings(data["dataset"])
        for w in warns:
            print(f"DATA NOTE {a.dataset}: {w}")
        for w in common.small_class_warnings(split):
            print(f"WARNING {a.dataset}: {w}; metrics on this split are very noisy")

        common.set_seed(init_seed, strict=a.strict_determinism)
        info = {"n_roi": int(data["corr"].shape[1]), "n_classes": data["n_classes"], "dataset_name": data["name"],
                "dataset_family": data["dataset"]}
        if hasattr(bl, "inputs"):
            data["inputs"] = bl.inputs(data, Path(a.out) / "_cache" / a.model, a.hp)
            info["n_time"] = int(data["inputs"][0].shape[-1])
        else:
            data["inputs"] = [data["corr"]]
        model, forward, groups = bl.build(info, a.hp, a.lr, a.weight_decay)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        cfg = dict(settings["train"])
        common.write_json(run_dir / "config.json", {
            "dataset": data["name"], "dataset_family": data["dataset"], "atlas": data["atlas"], "model": a.model,
            "run_seed": a.seed, "split_seed": split_seed, "init_seed": init_seed, "train": cfg, "base_lr": a.lr,
            "weight_decay": a.weight_decay, "model_hp": getattr(model, "hp_resolved", {**bl.DEFAULT_HP, **a.hp}), "n_params": n_params,
            "n_roi": info["n_roi"], "n_classes": data["n_classes"], "task": data["task"], "device": str(device),
            "git": common.git_info(), "env": common.env_info(), "argv": sys.argv, "data_file": data["file"],
            "tuned_from": a.tuned, "tuned_sha256": settings["tuned_sha256"], "tuned_seed": settings["tuned_seed"],
            "data_sha256": settings["data_sha256"], "fingerprint": fingerprint,
            "strict_determinism": bool(a.strict_determinism), "data_notes": warns,
            "label_names": {str(k): v for k, v in dataset_info.loaded_label_names(data["dataset"]).items()}})
        common.write_json(run_dir / "reproduce.json", {
            "command": " ".join([sys.executable] + sys.argv), "cwd": str(Path.cwd()), "data_file": data["file"],
            "data_sha256": settings["data_sha256"], "split_sha256": split["sha256"], "split_seed": split_seed,
            "init_seed": init_seed, "fingerprint": fingerprint, "git": common.git_info(), "env": common.env_info(),
            "determinism": "cudnn.deterministic, use_deterministic_algorithms(warn_only=" + str(not a.strict_determinism)
            + "), CUBLAS_WORKSPACE_CONFIG=:4096:8; seeds set by common.set_seed. GPU scatter ops in PyG layers have no "
            "deterministic kernel, so GPU reruns of graph models may differ in the last bits (use --strict_determinism to error)"})

        out = common.fit(model, forward, groups, data, idx, run_dir, cfg, device,
                         log=lambda s: print(f"[{a.dataset}/{a.model}/{a.seed}] {s}", flush=True))
        out.update(n_params=n_params, wall_seconds=time.time() - t0, split_sha256=split["sha256"], status="ok",
                   fingerprint=fingerprint)
        common.write_json(run_dir / "metrics.json", out)
        t = out["test"]
        print(f"DONE {a.dataset} {a.model} seed={a.seed} best_epoch={out['best_epoch']} test: auc={t['auc']:.4f} "
              f"acc={t['accuracy']:.4f} sens={t['sensitivity']:.4f} spec={t['specificity']:.4f}")
        return 0
    except Exception as e:  # recorded, never silently skipped
        rd.fail(e)
        print(f"FAILED {a.dataset} {a.model} seed={a.seed}: {e!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
