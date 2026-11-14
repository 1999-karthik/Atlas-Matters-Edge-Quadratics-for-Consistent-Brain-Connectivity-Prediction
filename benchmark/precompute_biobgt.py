#!/usr/bin/env python3
"""Compute and cache BioBGT per-subject features (VE, update_x) for every data file once, before the sweep.
Features use only each subject's own graph/time series (no labels, no split), so one cache serves all seeds.

    python benchmark/precompute_biobgt.py --data_dir Datasets --out runs --workers 16
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from adapters import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(common.REPO / "Datasets"))
    ap.add_argument("--out", default=str(common.REPO / "runs"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    bl = load("biobgt")
    for f in sorted(Path(a.data_dir).rglob("*.npy")):
        if a.only and not any(o.lower() in f.stem.lower() for o in a.only):
            continue
        t0 = time.time()
        d = common.load_dataset(f, dedup=False, with_timeseries=True)
        bl.inputs(d, Path(a.out) / "_cache" / "biobgt", workers=a.workers)
        print(f"{f.stem}: {len(d['labels'])} subjects, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
