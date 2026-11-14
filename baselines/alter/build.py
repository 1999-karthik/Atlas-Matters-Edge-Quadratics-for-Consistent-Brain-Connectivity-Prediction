"""Benchmark entry for this baseline: DEFAULT_HP, build(). See README.md for source and modifications."""
from omegaconf import OmegaConf

from .model import BrainNetworkTransformer

# sizes[0] is replaced by the ROI count inside the model (the first encoder layer does not pool); sizes[1] is the cluster count
DEFAULT_HP = dict(sizes=[360, 100], pooling=[False, True], pos_encoding="rrwp", pos_embed_dim=32,
                  dropout=0.1, ffn_dim=1024, activation="relu", encoder_hidden=32)  # dropout / ffn_dim: the upstream values, now tunable
FORWARD = lambda m, x: m(None, x)  # noqa: E731


def build(info, hp, base_lr, weight_decay):
    N = info["n_roi"]
    model_cfg = {**DEFAULT_HP, **hp}
    if "sizes" in model_cfg:
        model_cfg.update(orthogonal=True, freeze_center=True, project_assignment=True,
                         rrwp=dict(walk_length=8, threshold=0.0, add_identity=True))
    cfg = OmegaConf.create({"dataset": {"node_sz": N, "node_feature_sz": N, "num_classes": info["n_classes"]},
                            "model": model_cfg})
    model = BrainNetworkTransformer(cfg)
    groups = [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    return model, FORWARD, groups
