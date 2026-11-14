"""Benchmark entry for this baseline: DEFAULT_HP, build(). See README.md for source and modifications."""
from omegaconf import OmegaConf

from .model import BrainNetCNN

DEFAULT_HP = dict(dropout=0.5, width_mult=1.0, activation="leaky_relu")  # the upstream values (hard-coded there), now tunable
FORWARD = lambda m, x: m(None, x)  # noqa: E731


def build(info, hp, base_lr, weight_decay):
    N = info["n_roi"]
    model_cfg = {**DEFAULT_HP, **hp}
    cfg = OmegaConf.create({"dataset": {"node_sz": N, "node_feature_sz": N, "num_classes": info["n_classes"]},
                            "model": model_cfg})
    model = BrainNetCNN(cfg)
    groups = [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    return model, FORWARD, groups
