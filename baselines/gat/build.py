"""Benchmark entry for this baseline: DEFAULT_HP, build(). See README.md for source and modifications."""
from .model import ClassicGNN

# GNN recipe of the tunedGNN paper: residual connections + LayerNorm + dropout; hidden/layers/dropout are the
# tunable knobs (use --hp to override, tune on validation only).
DEFAULT_HP = dict(hidden=64, layers=3, dropout=0.5, res=True, ln=True, bn=False, jk=False, threshold=0.3, activation="relu", heads=4)
FORWARD = lambda m, x: m(x)  # noqa: E731


def build(info, hp, base_lr, weight_decay):
    model = ClassicGNN(info["n_roi"], info["n_classes"], gnn="gat", **{**DEFAULT_HP, **hp})
    groups = [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    return model, FORWARD, groups
