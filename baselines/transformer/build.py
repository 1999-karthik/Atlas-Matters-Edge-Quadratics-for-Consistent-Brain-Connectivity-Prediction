"""Benchmark entry for this baseline: DEFAULT_HP, build(). See README.md for source and modifications."""
from omegaconf import OmegaConf

from .model import GraphTransformer

DEFAULT_HP = dict(readout="mean", self_attention_layer=2, dropout=0.1, ffn_dim=1024, heads=4, activation="relu")  # the upstream values, now tunable
FORWARD = lambda m, x: m(None, x)  # noqa: E731


def build(info, hp, base_lr, weight_decay):
    N = info["n_roi"]
    model_cfg = {**DEFAULT_HP, **hp}
    cfg = OmegaConf.create({"dataset": {"node_sz": N, "node_feature_sz": N, "num_classes": info["n_classes"]},
                            "model": model_cfg})
    model = GraphTransformer(cfg)
    groups = [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    return model, FORWARD, groups
