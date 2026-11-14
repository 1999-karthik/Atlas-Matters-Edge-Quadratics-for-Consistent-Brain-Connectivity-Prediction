"""Benchmark entry for this baseline: DEFAULT_HP, build(). See README.md for source and modifications."""
from types import SimpleNamespace as NS

from .model import GCN

DEFAULT_HP = dict(hidden_dim=64, num_layers=3, dropout=0.5, pooling="mean", threshold=0.3, activation="relu")
FORWARD = lambda m, x: m(None, x)  # noqa: E731  (the model signature takes a time series it never reads)


def build(info, hp, base_lr, weight_decay):
    N = info["n_roi"]
    cfg = NS(dataset=NS(name=info["dataset_name"], node_sz=N, node_feature_sz=N, num_classes=info["n_classes"]),
             model=NS(**{**DEFAULT_HP, **hp}))
    model = GCN(cfg)
    groups = [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    return model, FORWARD, groups
