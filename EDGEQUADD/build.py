"""EDGEQUADD entry point for the benchmark runner: default hyper-parameters (DEFAULT_HP) and build()."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EDGEQUADD, make_param_groups, split_quad_group

DEFAULT_HP = dict(layers=3, dropout=0.2, activation="leaky_relu", pooling=True, cluster_num=4, droppath=0.05,
                  cnncorr_base_ch=32, cnncorr_norm="gn", cnncorr_diag_mode="zero", cnncorr_aspp_rates=[1, 2, 4], use_layer_norm=True, encoder_hidden=32,
                  quadratic_aspp_rates=[1, 2, 4], rank=3, frontend="conv1x1",
                  use_quad=True, use_aspp=True, use_gate=True,
                  alpha_init=0.0, quad_lr_mult=1.0, quad_input="graph", roi_perm_seed=None,
                  roi_ordering=None)  # None|identity|reverse|circular|random|block_swap_<k>


def _make_roi_perm(N, ordering, seed=42):
    """Return a numpy int64 permutation array for the given named ordering, or None for identity."""
    if ordering is None or ordering == "identity":
        return None
    rng = np.random.RandomState(seed)
    if ordering == "reverse":
        return np.arange(N - 1, -1, -1, dtype=np.int64)
    if ordering == "circular":
        return np.roll(np.arange(N, dtype=np.int64), N // 4)
    if ordering == "random":
        return rng.permutation(N).astype(np.int64)
    if ordering.startswith("block_swap"):
        # "block_swap_<k>" — permute k equal-sized blocks, ROIs within each block unchanged
        k = int(ordering.split("_")[-1]) if "_" in ordering[len("block_swap"):] else 8
        boundaries = np.linspace(0, N, k + 1, dtype=int)
        blocks = [np.arange(boundaries[i], boundaries[i + 1], dtype=np.int64) for i in range(k)]
        block_order = rng.permutation(k)
        return np.concatenate([blocks[i] for i in block_order])
    raise ValueError(f"unknown roi_ordering {ordering!r}; choose identity|reverse|circular|random|block_swap_<k>")


def build(info, hp, base_lr, weight_decay):
    if "adapter_lr_mult" in hp:
        raise ValueError("adapter_lr_mult is not supported: every parameter trains at the base learning rate. "
                         "Remove the key from the configuration.")
    hp = {**DEFAULT_HP, **hp}
    N = info["n_roi"]
    model = EDGEQUADD(args=NS(activation=hp["activation"]), node_sz=N, corr_pearson_sz=N, layers=hp["layers"],
                      dropout=hp["dropout"], cluster_num=hp["cluster_num"], pooling=hp["pooling"],
                      droppath=hp["droppath"], cnncorr_base_ch=hp["cnncorr_base_ch"], cnncorr_norm=hp["cnncorr_norm"],
                      cnncorr_diag_mode=hp["cnncorr_diag_mode"], cnncorr_aspp_rates=hp["cnncorr_aspp_rates"],
                      quadratic_aspp_rates=hp["quadratic_aspp_rates"], rank=hp["rank"], num_classes=info["n_classes"],
                      use_quad=hp["use_quad"], use_aspp=hp["use_aspp"], use_gate=hp["use_gate"],
                      use_cnncorr=hp["frontend"] != "none",
                      frontend=None if hp["frontend"] == "none" else hp["frontend"],
                      use_layer_norm=bool(hp["use_layer_norm"]), encoder_hidden=hp["encoder_hidden"])
    model.reset_learnable_gates()
    for a in model.adapters:
        a.alpha.data.fill_(hp["alpha_init"])
        a.quad_input = hp["quad_input"]
    groups = make_param_groups(model, lr=base_lr, weight_decay=weight_decay)
    if hp["quad_lr_mult"] != 1.0:
        groups = split_quad_group(model, groups, lr=base_lr * hp["quad_lr_mult"], weight_decay=0.0)
    fwd = (lambda m, x: m(x)) if hp["frontend"] == "none" else (lambda m, x: m(x, x))
    # ROI ordering: set roi_ordering (named ordering) or roi_perm_seed (random permutation), not both
    if hp.get("roi_ordering") is not None and hp.get("roi_perm_seed") is not None:
        raise ValueError("set roi_ordering OR roi_perm_seed, not both")
    perm_np = None
    if hp.get("roi_ordering") is not None:
        perm_np = _make_roi_perm(N, hp["roi_ordering"])
    elif hp.get("roi_perm_seed") is not None:
        perm_np = np.random.RandomState(int(hp["roi_perm_seed"])).permutation(N).astype(np.int64)
    if perm_np is not None:
        perm = torch.from_numpy(perm_np)
        inner = fwd
        fwd = lambda m, x: inner(m, x[:, perm.to(x.device)][:, :, perm.to(x.device)])
    return model, fwd, groups
