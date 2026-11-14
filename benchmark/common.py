"""Shared data, split, metric, training and artifact code for every model in the benchmark.

Protocol (identical for all models):
  * Input is the precomputed subject-level correlation matrix. No statistic is fitted across subjects, so
    no preprocessing step can see validation/test data or labels.
  * Split: two chained StratifiedShuffleSplit calls on the class label (70/10/20). The split depends only on
    (labels, split_seed); it never depends on the model, so every model sees the same subjects in a given run.
  * Seeds: `split_seed` controls the partition, `init_seed` controls weight init, batch order and mixup.
    Both default to the run seed (42, 1042, 2042, 3042, 4042).
  * Checkpoint selection: validation only (lowest validation loss by default). Test data is evaluated once,
    after selection.
  * Binary metrics: positive class = label 1 (patient), threshold p1 >= 0.5.
  * Multiclass (PPMI) metrics: macro one-vs-rest AUC from softmax scores; accuracy, macro recall and macro
    specificity from the argmax prediction.
"""
import copy
import functools
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 1042, 2042, 3042, 4042]
TRAIN_PROP, VAL_PROP = 0.7, 0.1
SPLIT_VERSION = 3  # version of the split procedure, recorded with every run


# ----------------------------------------------------------------------------- data
def parse_dataset_name(name):
    """'abide_AAL116' -> ('abide', 'AAL116'); 'adhd' -> ('adhd', 'adhd200_190roi')."""
    stem = name.lower()
    if stem == "adhd":
        return "adhd", "adhd200_190roi"
    ds, _, atlas = name.partition("_")
    return ds.lower(), atlas


def find_dataset_file(name, data_dir):
    """Locate <name>.npy under data_dir (searched recursively, case-insensitive)."""
    data_dir = Path(data_dir)
    for p in sorted(data_dir.rglob("*.npy")):
        if p.stem.lower() == name.lower():
            return p
    raise FileNotFoundError(f"{name}.npy not found under {data_dir}")


# ADHD-200 is used as released (459 scans, 54 of them exact duplicates of another scan; duplicates are kept in one
# partition). Every other dataset is deduplicated by default.
KEEP_DUPLICATES = {"adhd"}
# ADHD-200 files store control as 1; they are flipped on load so that label 1 = patient for every binary dataset.
FLIP_LABELS = {"adhd"}


def load_dataset(path, dedup=None, with_timeseries=False):
    """Load a dataset .npy into a dict with float32 corr [N,R,R], int64 labels [N], subject ids, metadata.

    Label convention: binary datasets load with 1 = patient (ASD / ADHD), 0 = control. ADHD-200 (stores control as 1) is
    flipped; ABIDE, ADNI and PPMI are loaded unchanged (label meanings in dataset_info.py).
    Exact duplicate correlation matrices are dropped (first occurrence kept) when dedup=True, because
    identical scans in different partitions would leak. dedup=None means "drop, except for datasets in KEEP_DUPLICATES";
    the duplicates found are always reported in `dup_info`.
    """
    if dedup is None:
        dedup = Path(path).stem.lower() not in KEEP_DUPLICATES
    d = np.load(path, allow_pickle=True).item()
    if "correlation" in d:
        corr, labels = d["correlation"], d["labels"]
    elif "adj" in d or "corr" in d:
        corr, labels = d.get("adj", d.get("corr")), d["label"]
    else:
        raise KeyError(f"{path}: unknown format, keys={list(d)}")
    corr = np.asarray(corr, dtype=np.float32)
    labels = np.asarray(labels).astype(np.int64)
    stem = Path(path).stem.lower()
    ds_name, atlas = parse_dataset_name(Path(path).stem)
    if ds_name in FLIP_LABELS:
        labels = 1 - labels
    ids = d.get("subject_ids")
    ids = [str(x) for x in ids] if ids is not None else [str(i) for i in range(len(labels))]
    if not (corr.ndim == 3 and corr.shape[1] == corr.shape[2] and len(corr) == len(labels) == len(ids)):
        raise ValueError(f"{path}: inconsistent shapes corr={corr.shape} labels={labels.shape}")
    if not np.isfinite(corr).all():
        raise ValueError(f"{path}: non-finite correlation values")
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(f"{path}: labels must be 0..K-1, got {classes}")

    dropped = []
    seen_ids = {}
    seen_hash = {}
    rep = np.arange(len(labels))  # rep[i] = first scan that i duplicates (itself if it is not a duplicate)
    for i in range(len(labels)):
        h = hashlib.md5(np.round(corr[i], 6).tobytes()).hexdigest()
        if ids[i] in seen_ids or h in seen_hash:
            dropped.append(i)
            rep[i] = seen_hash[h] if h in seen_hash else seen_ids[ids[i]]
        else:
            seen_ids[ids[i]] = i
            seen_hash[h] = i
    info = {"n_original": int(len(labels)), "n_duplicates": len(dropped), "duplicates_dropped": bool(dedup),
            "dropped_indices": dropped if dedup else []}
    keep = np.setdiff1d(np.arange(len(labels)), dropped) if dedup else np.arange(len(labels))
    extra = {}
    if with_timeseries:  # raw [S_orig, R, T], all subjects (some models cache per-file features over every subject)
        ts = np.asarray(d.get("timeseries", d.get("timeseires")), dtype=np.float32)
        orig_len = d.get("original_timeseries_lengths")  # valid length of each series (the rest of the row is zero padding)
        # files store [S, T, R]; some files store [S, R, T], recognised by its shape
        ts = ts if (ts.shape[1] == corr.shape[1] != ts.shape[2]) else ts.transpose(0, 2, 1)
        extra = {"raw_ts": np.ascontiguousarray(ts), "raw_corr": corr,
                 "raw_orig_len": None if orig_len is None else np.asarray(orig_len, dtype=np.int64)}
    return {
        **extra,
        "corr": corr[keep], "labels": labels[keep], "ids": [ids[i] for i in keep],
        "groups": rep[keep],  # scans sharing a value are duplicates of each other; make_split keeps them in one partition
        "orig_index": keep, "n_classes": int(len(classes)), "dataset": ds_name, "atlas": atlas,
        "name": Path(path).stem, "file": str(path), "dup_info": info,
        "task": "multiclass" if len(classes) > 2 else "binary",
    }


# ----------------------------------------------------------------------------- split
def make_split(labels, split_seed, train_prop=TRAIN_PROP, val_prop=VAL_PROP, groups=None):
    """Stratified train/val/test indices (positions in `labels`). Raises instead of falling back to a random split.

    `groups` (optional, one value per sample): samples with the same value are duplicates of one another and always land
    in the same partition, so a scan can never be trained on and tested on. The split is made over one representative per
    group (stratified by its label, sizes int(0.7 n), int(0.1 n), rest with n = number of groups) and each duplicate then
    follows its representative; partition sizes therefore differ from int(0.7 N), ... by up to the number of duplicates.
    With no duplicate the result is identical to a plain stratified split.
    """
    labels = np.asarray(labels)
    if groups is not None:
        groups = np.asarray(groups)
        _, first = np.unique(groups, return_index=True)
        reps = np.sort(first)  # representative of a group = its first member
        if len(reps) < len(labels):
            tr, va, te = make_split(labels[reps], split_seed, train_prop, val_prop)
            part = {}
            for name, ix in (("train", tr), ("val", va), ("test", te)):
                for r in reps[ix]:
                    part[groups[r]] = name
            out = [np.sort(np.array([i for i in range(len(labels)) if part[groups[i]] == name], dtype=int))
                   for name in ("train", "val", "test")]
            assert sum(len(o) for o in out) == len(labels) and not (set(out[0]) & set(out[1]) or set(out[0]) & set(out[2])
                                                                    or set(out[1]) & set(out[2]))
            return tuple(out)
    n = len(labels)
    n_train = int(n * train_prop)
    n_val = int(n * val_prop)
    n_test = n - n_train - n_val
    s1 = StratifiedShuffleSplit(n_splits=1, train_size=n_train, test_size=n - n_train, random_state=split_seed)
    train_idx, rest_idx = next(s1.split(np.zeros(n), labels))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=split_seed)
    v, t = next(s2.split(np.zeros(len(rest_idx)), labels[rest_idx]))
    val_idx, test_idx = rest_idx[v], rest_idx[t]
    train_idx, val_idx, test_idx = np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)
    assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx))
    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    return train_idx, val_idx, test_idx


def split_record(data, split_seed, idx):
    train_idx, val_idx, test_idx = idx
    ids, labels = data["ids"], data["labels"]
    groups = data.get("groups")
    n_groups = int(len(np.unique(groups))) if groups is not None else int(len(labels))
    rec = {"split_seed": split_seed, "n": int(len(labels)), "stratified_by": "label", "n_duplicate_groups_unit": n_groups,
           "duplicates_share_partition": bool(n_groups < len(labels)), "train_prop": TRAIN_PROP, "val_prop": VAL_PROP}
    for name, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        rec[name] = {"n": int(len(ix)), "indices": [int(i) for i in ix], "subject_ids": [ids[i] for i in ix],
                     "class_counts": {int(k): int(v) for k, v in zip(*np.unique(labels[ix], return_counts=True))}}
    rec["sha256"] = hashlib.sha256(json.dumps([rec["train"]["subject_ids"], rec["val"]["subject_ids"],
                                               rec["test"]["subject_ids"]]).encode()).hexdigest()
    return rec


# ----------------------------------------------------------------------------- metrics
def _safe_div(a, b):
    return float(a / b) if b > 0 else float("nan")


def binary_metrics(y_true, proba):
    """AUC (p1), accuracy, sensitivity (recall of class 1), specificity (recall of class 0); threshold p1 >= 0.5."""
    y_true = np.asarray(y_true)
    p1 = np.asarray(proba)[:, 1]
    pred = (p1 >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    auc = float(roc_auc_score(y_true, p1)) if len(np.unique(y_true)) == 2 else float("nan")
    sens, spec, prec = _safe_div(tp, tp + fn), _safe_div(tn, tn + fp), _safe_div(tp, tp + fp)
    return {"auc": auc, "accuracy": _safe_div(tp + tn, len(y_true)), "sensitivity": sens,
            "specificity": spec, "precision": prec, "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
            "balanced_accuracy": (sens + spec) / 2, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true))}


def multiclass_metrics(y_true, proba, n_classes):
    """Macro one-vs-rest AUC (softmax scores); accuracy / macro recall / macro specificity from argmax.

    Classes absent from y_true (no positives) are skipped in the macro averages; the number of classes
    averaged is reported so a reader can tell.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = proba.argmax(1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, pred):
        cm[t, p] += 1
    total = cm.sum()
    recalls, specs, aucs, precs, f1s, per_class = [], [], [], [], [], {}
    for k in range(n_classes):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        rec, spec = _safe_div(tp, tp + fn), _safe_div(tn, tn + fp)
        pos = (y_true == k)
        auc = float(roc_auc_score(pos.astype(int), proba[:, k])) if 0 < pos.sum() < len(pos) else float("nan")
        prec, f1 = _safe_div(tp, tp + fp), _safe_div(2 * tp, 2 * tp + fp + fn)
        per_class[k] = {"recall": rec, "specificity": spec, "precision": prec, "f1": f1, "auc": auc, "support": int(pos.sum())}
        recalls.append(rec); specs.append(spec); aucs.append(auc); precs.append(prec); f1s.append(f1)
    return {"auc": float(np.nanmean(aucs)), "accuracy": float(np.trace(cm) / total),
            "sensitivity": float(np.nanmean(recalls)), "specificity": float(np.nanmean(specs)),
            "precision": float(np.nanmean(precs)), "f1": float(np.nanmean(f1s)),
            "balanced_accuracy": float(np.nanmean(recalls)), "n_classes_averaged": int(np.sum(~np.isnan(recalls))), "confusion_matrix": cm.tolist(),
            "per_class": per_class, "n": int(len(y_true))}


def compute_metrics(y_true, proba, n_classes):
    return binary_metrics(y_true, proba) if n_classes == 2 else multiclass_metrics(y_true, proba, n_classes)


# ----------------------------------------------------------------------------- seeding / provenance
def set_seed(seed, strict=False):
    """Seed Python, NumPy and torch (CPU + CUDA) and ask for deterministic kernels.

    strict=False only warns when an operation has no deterministic implementation (PyG's scatter-add based layers on
    GPU are one such case, so GPU reruns can differ in the last bits); strict=True raises instead.
    PYTHONHASHSEED cannot be set from inside a running interpreter and nothing here depends on string hashing order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def file_sha256(path):
    st = os.stat(path)
    return _file_sha256(str(path), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=64)
def _file_sha256(path, size, mtime_ns):  # size and mtime are part of the cache key, so a changed file is re-hashed
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions():
    from importlib import metadata
    out = {}
    for pkg in ("torch", "numpy", "scipy", "scikit-learn", "networkx", "python-louvain", "torch_geometric", "torch_scatter",
                "torch_sparse", "einops", "pandas"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def git_info():
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    status = run("status", "--porcelain", "--", "EDGEQUADD", "baselines", "benchmark", "scripts")
    diff = run("diff", "HEAD", "--", "EDGEQUADD", "baselines", "benchmark", "scripts")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status),
            "diff_sha256": hashlib.sha256((diff + status).encode()).hexdigest() if status else None}


def env_info():
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "host": platform.node(), "packages": package_versions(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2)


# ----------------------------------------------------------------------------- training
def make_batches(n, batch_size, gen, shuffle=True):
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]
    # a trailing batch of one sample breaks BatchNorm in training; every other partial batch is kept
    return [b for b in batches if len(b) > 1] if shuffle else batches


@torch.no_grad()
def predict(model, forward, inputs, labels, batch_size, device):
    """Return (loss_mean, proba [n,K]) in eval mode. `inputs` is a list of per-subject tensors."""
    model.eval()
    probs, loss_sum = [], 0.0
    for i in range(0, len(labels), batch_size):
        xs = [t[i:i + batch_size].to(device) for t in inputs]
        y = labels[i:i + batch_size].to(device)
        logits = forward(model, *xs)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        probs.append(F.softmax(logits.float(), dim=1).cpu())
    return loss_sum / len(labels), torch.cat(probs).numpy()


def cosine_lr(base, step, total, final_ratio):
    return base * (final_ratio + (1 - final_ratio) * (1 + math.cos(math.pi * step / total)) / 2)


def scheduled_lr(cfg, base, epoch, it, steps_per_epoch, step, total_steps):
    """'cosine': per-step cosine from base to base*lr_final_ratio (BNT/ALTER/BQN/EDGEQUADD recipe).
    'warmup_cosine': BioBGT's upstream schedule - linear warm-up per iteration for `warmup_epochs`, then cosine
    annealing to 0 stepped once per epoch (torch CosineAnnealingLR with T_max = epochs - warmup)."""
    if cfg["schedule"] == "cosine":
        return cosine_lr(base, step, total_steps, cfg["lr_final_ratio"])
    w = cfg["warmup_epochs"]
    if epoch < w:
        return base * (epoch * steps_per_epoch + it) / (w * steps_per_epoch)
    return 0.5 * base * (1 + math.cos(math.pi * (epoch - w) / max(1, cfg["epochs"] - w)))


def fit(model, forward, param_groups, data, idx, run_dir, cfg, device, log=print, on_epoch=None):
    """Train for cfg['epochs'] epochs; select the epoch on validation only; evaluate test once with that checkpoint.

    cfg keys: epochs, batch_size, optimizer ('adam'|'adamw'), schedule ('cosine'|'warmup_cosine'), warmup_epochs, lr_final_ratio, grad_clip (0=off), mixup ('hard'|'soft'|'none'), selection ('val_loss'|'val_auc'),
              init_seed; optional skip_test (tuning: never touch test data, write no checkpoint/predictions).
    on_epoch(epoch, val_loss), if given, is called after every epoch (used by the tuner for pruning).
    """
    train_idx, val_idx, test_idx = idx
    K = data["n_classes"]
    inputs = [torch.from_numpy(np.ascontiguousarray(a)) for a in data["inputs"]]
    labels = torch.from_numpy(data["labels"])
    if cfg["mixup"] != "none" and len(inputs) > 1:
        raise ValueError("mixup is only defined for single-input models")
    c_tr, y_tr = [t[train_idx].to(device) for t in inputs], labels[train_idx].to(device)
    c_va, y_va = [t[val_idx] for t in inputs], labels[val_idx]
    c_te, y_te = [t[test_idx] for t in inputs], labels[test_idx]

    optimizer = (torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam)(param_groups)
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]
    gen = torch.Generator().manual_seed(cfg["init_seed"])
    mix_rng = np.random.RandomState(cfg["init_seed"])
    epochs, bs = cfg["epochs"], cfg["batch_size"]
    steps_per_epoch = len(make_batches(len(train_idx), bs, torch.Generator().manual_seed(0)))
    total_steps = max(1, steps_per_epoch * epochs)
    step = 0
    best = {"score": None, "epoch": -1, "state": None}
    log_path = Path(run_dir) / "train_log.jsonl"
    t0 = time.time()

    with open(log_path, "w") as logf:
        for epoch in range(epochs):
            model.train()
            loss_sum, n_seen = 0.0, 0
            for it, b in enumerate(make_batches(len(train_idx), bs, gen)):
                b = b.to(device)
                xs, y = [t[b] for t in c_tr], y_tr[b]
                for g in optimizer.param_groups:
                    g["lr"] = scheduled_lr(cfg, g["base_lr"], epoch, it, steps_per_epoch, step, total_steps)
                step += 1
                if cfg["mixup"] != "none":
                    lam = float(mix_rng.beta(1.0, 1.0))
                    perm = torch.from_numpy(mix_rng.permutation(len(b))).to(device)
                    x = xs[0]
                    logits = forward(model, lam * x + (1 - lam) * x[perm])
                    if cfg["mixup"] == "hard":  # original behaviour: CE on argmax of the mixed one-hot label
                        y_hard = y if lam >= 0.5 else y[perm]
                        loss = F.cross_entropy(logits, y_hard, reduction="sum")
                    else:
                        loss = lam * F.cross_entropy(logits, y, reduction="sum") + \
                            (1 - lam) * F.cross_entropy(logits, y[perm], reduction="sum")
                else:
                    logits = forward(model, *xs)
                    loss = F.cross_entropy(logits, y, reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}, step {step}")
                optimizer.zero_grad(set_to_none=True)
                (loss / len(b)).backward()
                if cfg["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                loss_sum += loss.item()
                n_seen += len(b)

            val_loss, val_p = predict(model, forward, c_va, y_va, bs, device)
            val_m = compute_metrics(y_va.numpy(), val_p, K)
            score = -val_loss if cfg["selection"] == "val_loss" else val_m["auc"] - 1e-6 * val_loss
            if best["score"] is None or score > best["score"]:
                best = {"score": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            rec = {"epoch": epoch, "train_loss": loss_sum / max(1, n_seen), "val_loss": val_loss,
                   "val_auc": val_m["auc"], "val_accuracy": val_m["accuracy"], "lr": optimizer.param_groups[0]["lr"],
                   "elapsed_s": round(time.time() - t0, 1)}
            logf.write(json.dumps(_jsonable(rec)) + "\n"); logf.flush()
            if on_epoch is not None:
                on_epoch(epoch, val_loss)
            if epoch % 10 == 0 or epoch == epochs - 1:
                log(f"epoch {epoch}/{epochs} train_loss={rec['train_loss']:.4f} val_loss={val_loss:.4f} "
                    f"val_auc={val_m['auc']:.4f}")

    model.load_state_dict(best["state"])
    if cfg.get("skip_test"):
        return {"best_epoch": best["epoch"], "best_val_loss": -best["score"] if cfg["selection"] == "val_loss" else None,
                "train_seconds": time.time() - t0}
    torch.save({"state_dict": best["state"], "epoch": best["epoch"]}, Path(run_dir) / "best.pt")
    out = {"best_epoch": best["epoch"], "selection": cfg["selection"], "train_seconds": time.time() - t0}
    preds = {}
    for name, c, y, ix in (("val", c_va, y_va, val_idx), ("test", c_te, y_te, test_idx)):
        loss, p = predict(model, forward, c, y, bs, device)
        out[name] = {**compute_metrics(y.numpy(), p, K), "loss": loss}
        preds[name] = (ix, p)
    for name in ("val", "test"):  # test_predictions.csv holds test rows only
        write_predictions(Path(run_dir) / f"{name}_predictions.csv", data, {name: preds[name]}, K)
    return out


def write_predictions(path, data, preds, K):
    with open(path, "w") as f:
        f.write("split,subject_index,subject_id,y_true," + ",".join(f"p{k}" for k in range(K)) + ",y_pred\n")
        for name, (ix, p) in preds.items():
            for j, i in enumerate(ix):
                f.write(f"{name},{int(data['orig_index'][i])},{data['ids'][i]},{int(data['labels'][i])},"
                        + ",".join(f"{v:.8f}" for v in p[j]) + f",{int(p[j].argmax())}\n")


def run_path(root, dataset_name, model, seed):
    """<root>/<model>/<dataset family>/<atlas>/seed_<seed>, e.g. runs/gcn/abide/AAL116/seed_42."""
    family, atlas = parse_dataset_name(dataset_name)
    return Path(root) / model / family / atlas / f"seed_{seed}"


ARTEFACTS = ("FAILED.json", "metrics.json", "best.pt", "val_predictions.csv", "test_predictions.csv", "train_log.jsonl",
             "config.json", "split.json", "reproduce.json", "run.log")


def settings_fingerprint(settings):
    """sha256 of everything that determines a run's result except the code: dataset file, split, model, training and
    model hyper-parameters, tuned file. A finished run is only reused when this matches."""
    return hashlib.sha256(json.dumps(_jsonable(settings), sort_keys=True).encode()).hexdigest()


class RunDir:
    """Creates <root>/<model>/<dataset family>/<atlas>/seed_<s>/ and records status."""

    def __init__(self, root, dataset_name, model, seed):
        self.path = run_path(root, dataset_name, model, seed)

    def done(self, fingerprint=None):
        """True if a finished run exists (and, when `fingerprint` is given, was produced with the same settings)."""
        m = self.path / "metrics.json"
        if not m.exists():
            return False
        if fingerprint is None:
            return True
        try:
            return json.load(open(m)).get("fingerprint") == fingerprint
        except (OSError, ValueError):
            return False

    def fresh(self):
        """Create the folder and remove every artefact of an earlier attempt, so a run never mixes old and new files."""
        self.path.mkdir(parents=True, exist_ok=True)
        for f in ARTEFACTS:
            (self.path / f).unlink(missing_ok=True)
        return self.path

    def fail(self, exc):
        write_json(self.path / "FAILED.json", {"error": repr(exc), "traceback": traceback.format_exc()})


def small_class_warnings(split, min_n=5):
    """Messages for classes with fewer than `min_n` subjects in validation or test (their metrics are very noisy)."""
    out = []
    for part in ("val", "test"):
        for k, n in split[part]["class_counts"].items():
            if n < min_n:
                out.append(f"class {k} has only {n} subject(s) in {part}")
    return out
