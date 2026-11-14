"""Hyper-parameter search spaces for the tuning protocol (benchmark/tune.py).

Fairness rules
  * Every model gets the SAME training space (`suggest_train`): lr, weight decay, batch size, epochs, optimiser,
    schedule, gradient clipping and mixup.
  * Every model also gets dropout with one shared range (0.1-0.9), every model that has a depth gets the same range (1-6), every model gets the same four activations for its main blocks, and
    every model with a width gets a width knob. Models differ only where their code has no such knob (BrainNetCNN has no depth,
    BQN no width, BNT / ALTER a fixed two blocks) or has extra ones of its own (graph threshold, residual / norm switches,
    EDGEQUADD's rank, ...). Knobs that were hard-coded upstream (dropout of BNT / ALTER / Transformer / BrainNetCNN, FFN width,
    BioBGT's FFN multiple) are exposed with the upstream value as default, so trial 0 is the unchanged upstream network.
  * Trial 0 is the shared-recipe default and, where it differs, the authors' own configuration is enqueued as trial 1
    (`default_trials`), so the authors' setting is always one of the evaluated points.
  * Every study uses the same sampler and pruner; the trial budget is set with --trials. Trials are selected on the
    validation set only; test data is never read while tuning.
"""
import os

TRAIN_DEFAULT = dict(lr=1e-4, weight_decay=1e-4, batch_size=16, epochs=200, optimizer="adam", schedule="cosine",
                     warmup_epochs=10, lr_final_ratio=0.1, grad_clip=0.0, mixup="soft")

NO_MIXUP = {"biobgt"}  # three input tensors (time series, VE, update_x): mixup is undefined, so it is fixed to "none"
ENCODER_HIDDEN = [16, 32, 64, 128]  # width of the clustering encoder (BNT, ALTER, BQN, EDGEQUADD)
ASPP_RATES = {"1-2-4": [1, 2, 4], "1-2": [1, 2], "1-3-6": [1, 3, 6], "1-2-4-8": [1, 2, 4, 8]}  # dilation rates of EDGEQUADD's ASPP blocks
ACTIVATIONS = ["relu", "gelu", "elu", "leaky_relu"]  # activation of each model's main blocks (not of every MLP head)


FRONTENDS = ["none", "conv1x1", "aspp", "cnn"]  # EDGEQUADD
HEAVY_FRONTENDS = ("aspp", "cnn")  # their activations scale with batch x regions^2 x channels: small batches only


def _batches(sizes):
    """Batch sizes, optionally restricted to >= $TUNE_MIN_BATCH (a study-wide restriction; unset = the full list)."""
    lo = int(os.environ.get("TUNE_MIN_BATCH", 0))
    return [b for b in sizes if b >= lo] or [max(sizes)]


def _epochs(choices):
    """Epoch choices, optionally filtered by $TUNE_MIN_EPOCHS / $TUNE_MAX_EPOCHS."""
    lo = int(os.environ.get("TUNE_MIN_EPOCHS", 0))
    hi = int(os.environ.get("TUNE_MAX_EPOCHS", 10**9))
    return [e for e in choices if lo <= e <= hi] or [min(choices)]


def suggest_train(trial, model):
    # EDGEQUADD's aspp / cnn front-ends need far more memory per sample; with them the batch size is drawn from a short list
    heavy = model == "edgequadd" and trial.suggest_categorical("frontend", FRONTENDS) in HEAVY_FRONTENDS
    _min_wd = float(os.environ.get("TUNE_MIN_WD", 1e-7))
    t = {
        "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", _min_wd, 1e-1, log=True),
        "batch_size": (trial.suggest_categorical("batch_size_heavy", _batches([4, 8, 16])) if heavy
                       else trial.suggest_categorical("batch_size", _batches([4, 8, 16, 32, 64, 128]))),
        "epochs": trial.suggest_categorical("epochs", _epochs([50, 100, 200, 300, 400])),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "schedule": trial.suggest_categorical("schedule", ["cosine", "warmup_cosine"]),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.0, 1.0, 5.0]),
    }
    t["warmup_epochs"] = trial.suggest_categorical("warmup_epochs", [5, 10]) if t["schedule"] == "warmup_cosine" else 0
    # the final lr only exists for the plain cosine schedule (warmup_cosine anneals to 0)
    t["lr_final_ratio"] = trial.suggest_categorical("lr_final_ratio", [0.01, 0.1, 0.5, 1.0]) if t["schedule"] == "cosine" else 0.1
    t["mixup"] = "none" if model in NO_MIXUP else trial.suggest_categorical("mixup", ["none", "soft", "hard"])  # hard: CE on the argmax of the mixed label
    return t


DROPOUT = dict(low=0.1, high=0.9, step=0.1)  # one range for every model (0.0 is not sampled)
DEPTH = (1, 6)                                # one range for every model that has a depth
WIDTHS = [16, 32, 64, 128, 256, 512]            # hidden / channel width of the graph models
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # graph models: edge iff corr > threshold
FFN_DIMS = [128, 256, 512, 1024, 2048, 4096]             # feed-forward width of the transformer blocks


def _activation(trial):
    return trial.suggest_categorical("activation", ACTIVATIONS)


def _dropout(trial):
    return trial.suggest_float("dropout", **DROPOUT)


def _depth(trial, name="layers"):
    return trial.suggest_int(name, *DEPTH)


def _tuned_gnn(trial, heads=(1,)):
    return dict(hidden=trial.suggest_categorical("hidden", WIDTHS), layers=_depth(trial), dropout=_dropout(trial), activation=_activation(trial),
                res=trial.suggest_categorical("res", [True, False]), ln=trial.suggest_categorical("ln", [True, False]),
                bn=trial.suggest_categorical("bn", [True, False]), jk=trial.suggest_categorical("jk", [True, False]),
                threshold=trial.suggest_categorical("threshold", THRESHOLDS),
                heads=trial.suggest_categorical("heads", list(heads)) if len(heads) > 1 else heads[0])


def _edgequadd_frontend(trial):
    """EDGEQUADD's front-end and only the settings of the front-end that was chosen (the others would be dead knobs).
    conv1x1 = 1x1 convolution; aspp = multi-scale ASPP front-end; cnn = convolutional extractor (SimpleCNNCorrExtractor)."""
    hp = {"frontend": trial.suggest_categorical("frontend", FRONTENDS)}
    if hp["frontend"] in ("aspp", "cnn"):
        hp["cnncorr_aspp_rates"] = ASPP_RATES[trial.suggest_categorical("cnn_aspp", list(ASPP_RATES))]
    if hp["frontend"] == "cnn":
        hp.update(cnncorr_base_ch=trial.suggest_categorical("base_ch", [16, 32, 64, 128]),
                  cnncorr_norm=trial.suggest_categorical("cnncorr_norm", ["gn", "bn"]),
                  cnncorr_diag_mode=trial.suggest_categorical("cnncorr_diag_mode", ["zero", "keep", "channel"]))
    return hp


def suggest_model(trial, model, info):
    """Architecture search space of `model`. Every model gets dropout (same range), its depth (same range) and a width knob
    wherever its code has one; knobs a model does not have (BrainNetCNN depth, BQN width, ...) are simply absent."""
    if model in ("gcn", "graphsage"):
        return _tuned_gnn(trial)
    if model == "gat":
        return _tuned_gnn(trial, heads=(1, 2, 4, 8))
    if model == "gcn_plain":
        return dict(hidden_dim=trial.suggest_categorical("hidden", WIDTHS), num_layers=_depth(trial), activation=_activation(trial),
                    dropout=_dropout(trial), threshold=trial.suggest_categorical("threshold", THRESHOLDS))
    if model == "gps":
        channels = trial.suggest_categorical("channels", WIDTHS)
        return dict(channels=channels, pe_dim=int(channels * trial.suggest_categorical("pe_frac", [0.25, 0.5])),
                    num_layers=_depth(trial), dropout=_dropout(trial), heads=trial.suggest_categorical("heads", [2, 4, 8]),
                    walk_length=trial.suggest_categorical("walk_length", [4, 8, 16, 32]), activation=_activation(trial),
                    threshold=trial.suggest_categorical("threshold", THRESHOLDS),
                    use_positional_encoding=trial.suggest_categorical("use_pe", [True, False]))
    if model in ("bnt", "alter"):
        # sizes[0] is overwritten by the number of ROIs inside the model (first layer does not pool); only the cluster count
        # of the second, pooling layer is a real choice. Two transformer blocks, as upstream, is not a knob.
        hp = dict(sizes=[360, trial.suggest_categorical("clusters", [100, 50, 25, 10])], dropout=_dropout(trial), activation=_activation(trial),
                  encoder_hidden=trial.suggest_categorical("encoder_hidden", ENCODER_HIDDEN),
                  ffn_dim=trial.suggest_categorical("ffn_dim", FFN_DIMS))
        if model == "alter":
            hp["pos_embed_dim"] = trial.suggest_categorical("pos_embed_dim", [16, 32, 64])
        return hp
    if model == "transformer":
        return dict(self_attention_layer=_depth(trial), dropout=_dropout(trial), ffn_dim=trial.suggest_categorical("ffn_dim", FFN_DIMS),
                    heads=trial.suggest_categorical("heads", [2, 4, 8]), activation=_activation(trial))  # heads: preferred count; the largest divisor of the ROI count <= it
    if model == "brainnetcnn":
        # fixed depth (E2E, E2N, N2G, 3 dense); `width_mult` scales all its channel / unit counts (upstream = 1.0)
        return dict(dropout=_dropout(trial), width_mult=trial.suggest_categorical("width_mult", [0.25, 0.5, 1.0, 2.0]), activation=_activation(trial))
    if model == "bqn":  # no width knob: the quadratic layers are as wide as the ROI count
        return dict(layers=_depth(trial), dropout=_dropout(trial), cluster_num=trial.suggest_categorical("cluster_num", [2, 4, 8, 16]),
                    activation=_activation(trial), encoder_hidden=trial.suggest_categorical("encoder_hidden", ENCODER_HIDDEN))
    if model == "edgequadd":
        return dict(layers=_depth(trial), dropout=_dropout(trial), droppath=trial.suggest_categorical("droppath", [0.0, 0.05, 0.1, 0.2, 0.3]),
                    cluster_num=trial.suggest_categorical("cluster_num", [2, 4, 8, 16, 32]), activation=_activation(trial),
                    rank=trial.suggest_categorical("rank", [1, 2, 3, 4, 6, 8, 12, 16]),
                    quad_lr_mult=trial.suggest_categorical("quad_lr_mult", [0.25, 0.5, 1.0, 2.0, 4.0]),
                    alpha_init=trial.suggest_categorical("alpha_init", [0.0, 0.5, 1.0]),
                    encoder_hidden=trial.suggest_categorical("encoder_hidden", ENCODER_HIDDEN),
                    use_layer_norm=trial.suggest_categorical("use_layer_norm", [True, False]),
                    quadratic_aspp_rates=ASPP_RATES[trial.suggest_categorical("quad_aspp", list(ASPP_RATES))],
                    **_edgequadd_frontend(trial))
    if model == "biobgt":
        return dict(d_model=trial.suggest_categorical("d_model", [32, 64, 128, 256, 512]),  # update_x stays 128-wide (extractor)
                    num_heads=trial.suggest_categorical("heads", [2, 4, 8, 16]), num_layers=_depth(trial), dropout=_dropout(trial),
                    ffn_mult=trial.suggest_categorical("ffn_mult", [1, 2, 4, 8]), activation=_activation(trial))
    raise KeyError(model)


def default_trials(model, info):
    """Parameter dicts (by suggest-name) enqueued first: shared-recipe default, then the authors' own configuration."""
    t = dict(TRAIN_DEFAULT)
    if model in NO_MIXUP:
        t.pop("mixup")
    if t["schedule"] != "warmup_cosine":
        t.pop("warmup_epochs")
    if t["schedule"] != "cosine":
        t.pop("lr_final_ratio")
    drop = 0.5 if info.get("dataset_family", "").lower() == "abide" else 0.1
    m = {
        "gcn": dict(hidden=64, layers=3, dropout=0.5, res=True, ln=True, bn=False, jk=False, threshold=0.3, activation="relu"),
        "graphsage": dict(hidden=64, layers=3, dropout=0.5, res=True, ln=True, bn=False, jk=False, threshold=0.3, activation="relu"),
        "gat": dict(hidden=64, layers=3, dropout=0.5, res=True, ln=True, bn=False, jk=False, threshold=0.3, heads=4, activation="relu"),
        "gcn_plain": dict(hidden=64, layers=3, dropout=0.5, threshold=0.3, activation="relu"),
        "gps": dict(channels=64, pe_frac=0.5, layers=3, dropout=0.5, heads=4, walk_length=8, threshold=0.3, use_pe=True, activation="relu"),
        "bnt": dict(clusters=100, dropout=0.1, ffn_dim=1024, activation="relu", encoder_hidden=32),
        "alter": dict(clusters=100, dropout=0.1, ffn_dim=1024, pos_embed_dim=32, activation="relu", encoder_hidden=32),
        "transformer": dict(layers=2, dropout=0.1, ffn_dim=1024, heads=4, activation="relu"),
        "brainnetcnn": dict(dropout=0.5, width_mult=1.0, activation="leaky_relu"),
        "bqn": dict(layers=3, dropout=0.1, cluster_num=4, activation="leaky_relu", encoder_hidden=32),
        "edgequadd": dict(layers=3, dropout=0.2, droppath=0.05, cluster_num=4, rank=3, quad_lr_mult=1.0, alpha_init=1.0,
                      activation="leaky_relu", encoder_hidden=32, frontend="conv1x1", use_layer_norm=True, quad_aspp="1-2-4"),
        "biobgt": dict(d_model=128, heads=8, layers=3, dropout=drop, ffn_mult=2, activation="relu"),
    }[model]
    out = [{**t, **m}]
    if model == "biobgt":  # authors' recipe: AdamW 3e-4, wd 1e-4, 10 warm-up epochs + cosine, batch 32
        out.append({**out[0], "optimizer": "adamw", "lr": 3e-4, "batch_size": 32, "schedule": "warmup_cosine",
                    "warmup_epochs": 10})
    return out
