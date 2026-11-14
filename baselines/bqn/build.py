"""Benchmark entry for this baseline: DEFAULT_HP, build(). See README.md for source and modifications."""
from types import SimpleNamespace as NS

from .model import Quadratic_BN

DEFAULT_HP = dict(layers=3, dropout=0.1, activation="leaky_relu", encoder_hidden=32, pooling=True, cluster_num=4)
FORWARD = lambda m, x: m(None, x)  # noqa: E731


def build(info, hp, base_lr, weight_decay):
    hp = {**DEFAULT_HP, **hp}
    N = info["n_roi"]
    model = Quadratic_BN(NS(activation=hp["activation"]), node_sz=N, time_series_sz=0, corr_pearson_sz=N,
                         layers=hp["layers"], dropout=hp["dropout"], cluster_num=hp["cluster_num"],
                         pooling=hp["pooling"], num_classes=info["n_classes"], encoder_hidden=hp["encoder_hidden"])
    groups = [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    return model, FORWARD, groups
