"""Benchmark entry for BioBGT. Defaults are the paper's Table 3 / the authors' released training script."""
import os
from pathlib import Path

from .model import BioBGT

# Paper: 3 layers, 8 heads, hidden 128 (FFN 256 = 2 x hidden), dropout 0.5 on ABIDE and 0.1 on ADNI / ADHD-200
# (PPMI is not in the paper: 0.1). `dropout=None` selects that per-dataset value.
DEFAULT_HP = dict(d_model=128, num_heads=8, num_layers=3, dropout=None, ffn_mult=2, activation="relu", extractor_variant="code", series_mode=None)
# Trains with the shared benchmark recipe. The authors' recipe (AdamW 3e-4, wd 1e-4, 10 warm-up epochs + cosine, batch 32) is
# evaluated as one of the first tuning trials (benchmark/spaces.py). Mixup is undefined for three input tensors, so it is off.
TRAIN_OVERRIDES = dict(mixup="none")
NEEDS_TIMESERIES = True
FORWARD = lambda m, ts, ve, ux: m(ts, ve, ux)  # noqa: E731


def inputs(data, cache_dir, hp=None, workers=8):
    """[prepared time series (see features.prepare_series), VE, update_x] for the kept subjects. Per subject, label-free, cached per
    data file (keyed by file hash, edge threshold and extractor variant)."""
    from .features import graph_threshold, load_or_compute, prepare_series
    hp = {**DEFAULT_HP, **(hp or {})}
    mode = hp.get("series_mode") or os.environ.get("BIOBGT_SERIES", "tile")  # tune.py passes no hp: the env var reaches it
    ts = prepare_series(data["raw_ts"], data.get("raw_orig_len"), mode=mode)
    if mode != "tile":
        cache_dir = Path(cache_dir) / mode  # the cache file is named after the data file only: keep the modes apart
    ve, ux = load_or_compute(data["file"], ts, data["raw_corr"], cache_dir,
                             threshold=graph_threshold(data["dataset"]), variant=hp["extractor_variant"], workers=workers)
    idx = data["orig_index"]
    return [ts[idx], ve[idx], ux[idx]]


def build(info, hp, base_lr, weight_decay):
    hp = {**DEFAULT_HP, **hp}
    hp.pop("extractor_variant")
    hp.pop("series_mode", None)
    if hp["dropout"] is None:
        hp["dropout"] = 0.5 if info.get("dataset_family", "").lower() == "abide" else 0.1
    model = BioBGT(in_size=info["n_time"], num_classes=info["n_classes"], **hp)
    model.hp_resolved = dict(hp)
    groups = [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    return model, FORWARD, groups
