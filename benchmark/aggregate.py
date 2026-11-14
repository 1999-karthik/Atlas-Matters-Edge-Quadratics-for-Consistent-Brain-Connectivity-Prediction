#!/usr/bin/env python3
"""Aggregate finished runs. Metrics are recomputed from test_predictions.csv (not trusted from metrics.json) and
compared against metrics.json. Writes <runs>/summary.csv, <runs>/per_run.csv, <runs>/status.json and prints a table.

    python benchmark/aggregate.py --runs runs
Mean +- sample std (ddof=1) over the 5 seeds; values are percentages.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from adapters import MODELS  # noqa: E402


METRICS = ("auc", "accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy")


def recompute(pred_csv, n_classes):
    rows = [r for r in csv.DictReader(open(pred_csv)) if r["split"] == "test"]
    y = np.array([int(r["y_true"]) for r in rows])
    p = np.array([[float(r[f"p{k}"]) for k in range(n_classes)] for r in rows])
    return common.compute_metrics(y, p, n_classes)


def discover(root):
    """(dataset, model, seed) of every run folder under root, used when there is no jobs.tsv."""
    found = set()
    for rd in sorted(Path(root).glob("*/*/*/seed_*")):
        if not rd.is_dir() or rd.parts[-4].startswith("_"):
            continue
        model, seed = rd.parts[-4], int(rd.name.split("_")[1])
        if (rd / "config.json").exists():
            dataset = json.load(open(rd / "config.json"))["dataset"]
        else:  # failed before config.json: rebuild the name from the folder names (<family>/<atlas>)
            family, atlas = rd.parts[-3], rd.parts[-2]
            dataset = family if family == "adhd" else f"{family}_{atlas}"
        found.add((dataset, model, seed))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(common.REPO / "runs"))
    ap.add_argument("--seeds", nargs="+", type=int, default=common.SEEDS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--jobs", default=None, help="tsv with the planned runs (default: <runs>/jobs.tsv)")
    a = ap.parse_args()
    root = Path(a.runs)
    jobs_file = Path(a.jobs) if a.jobs else root / "jobs.tsv"
    jobs = [l.split("\t") for l in jobs_file.read_text().splitlines()[1:]] if jobs_file.exists() else []
    planned = {(d, m, int(s)) for d, m, s in jobs}
    if not planned:
        planned = discover(root)
        print(f"no {root / 'jobs.tsv'}: aggregating the {len(planned)} run folders found (missing runs cannot be reported)")
    if not planned:
        sys.exit(f"nothing to aggregate under {root}")
    per_run, status, mismatches = [], {"ok": [], "failed": [], "missing": []}, []
    for d, m, s in sorted(planned):
        rd = common.run_path(root, d, m, s)
        if (rd / "metrics.json").exists():
            cfg = json.load(open(rd / "config.json"))
            mj = json.load(open(rd / "metrics.json"))
            t = recompute(rd / "test_predictions.csv", cfg["n_classes"])
            for k in METRICS:
                if k not in mj["test"]:  # metric not recorded for this run
                    continue
                ref = mj["test"][k]
                if (ref is None) != (not np.isfinite(t[k])) or (ref is not None and abs(ref - t[k]) > 1e-6):
                    mismatches.append((d, m, s, k, ref, t[k]))
            per_run.append({"dataset": d, "model": m, "seed": s, "best_epoch": mj["best_epoch"], "task": cfg["task"],
                            **{k: t[k] for k in METRICS}})
            status["ok"].append([d, m, s])
        elif (rd / "FAILED.json").exists():
            status["failed"].append([d, m, s, json.load(open(rd / "FAILED.json"))["error"]])
        else:
            status["missing"].append([d, m, s])
    with open(root / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "seed", "task", "best_epoch", *METRICS])
        w.writeheader(); w.writerows(per_run)
    groups = defaultdict(list)
    for r in per_run:
        groups[(r["dataset"], r["model"])].append(r)
    for (d, m), rs in groups.items():  # per-folder record: runs/<model>/<family>/<atlas>/summary.json
        folder = common.run_path(root, d, m, 0).parent
        common.write_json(folder / "summary.json", {
            "dataset": d, "model": m, "seeds": [r["seed"] for r in rs], "best_epochs": [r["best_epoch"] for r in rs],
            **{k: {"per_seed": [r[k] for r in rs], "mean": float(np.mean([r[k] for r in rs])),
                   "std_ddof1": float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else None}
               for k in METRICS}})
        with open(folder / "metrics.csv", "w", newline="") as f:  # per-seed test metrics for this model/dataset/atlas
            w = csv.writer(f); w.writerow(["seed", "best_epoch", *METRICS])
            for r in rs:
                w.writerow([r["seed"], r["best_epoch"], *[r[k] for k in METRICS]])
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "n_seeds"] + [f"{k}_{s}" for k in METRICS for s in ("mean", "std")] + ["complete"])
        for (d, m), rs in sorted(groups.items()):
            row = [d, m, len(rs)]
            for k in METRICS:
                v = np.array([r[k] for r in rs]) * 100
                row += [round(float(np.nanmean(v)), 2), round(float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"), 2)]
            row.append(len(rs) == len(a.seeds))
            w.writerow(row)
    status["metric_mismatches"] = mismatches
    json.dump(status, open(root / "status.json", "w"), indent=1)
    print(f"planned={len(planned)} ok={len(status['ok'])} failed={len(status['failed'])} missing={len(status['missing'])} "
          f"mismatches={len(mismatches)}")
    print((root / "summary.csv").read_text())


if __name__ == "__main__":
    main()
