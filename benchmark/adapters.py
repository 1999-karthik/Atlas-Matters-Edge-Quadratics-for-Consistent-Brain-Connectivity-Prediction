"""Model registry. Every baseline lives in its own folder (`baselines/<name>/`, EDGEQUADD in `EDGEQUADD/`) with
`model.py`, `build.py` and `README.md`; this module loads `build.py` by path under a unique package name, so
folders never collide even though several use the same internal module names.

build.py contract:
    DEFAULT_HP: dict                       model hyper-parameters (overridable with run.py --hp)
    build(info, hp, base_lr, weight_decay) -> (model, forward, param_groups)
        info = {n_roi, n_classes, dataset_name, n_time (only if inputs() exists)}; forward(model, *batch_tensors) -> logits
    optional  inputs(data, cache_dir) -> list of per-subject arrays (default: [correlation matrices])
    optional  TRAIN_OVERRIDES: dict        e.g. {"mixup": "none"}
    optional  NEEDS_TIMESERIES: bool       load raw time series into `data`
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = ["edgequadd", "gcn_plain", "gps", "gcn", "graphsage", "gat", "brainnetcnn", "bnt", "alter",
          "transformer", "bqn", "biobgt"]
FOLDER = {m: REPO / "baselines" / m for m in MODELS}
FOLDER["edgequadd"] = REPO / "EDGEQUADD"
FOLDER["gcn_plain"] = REPO / "baselines" / "gps" / "gcn_plain"  # in-house GCN lives next to GPS


def load(name):
    folder = FOLDER[name]
    pkg = f"bl_{name}"
    if (folder / "__init__.py").exists():
        spec = importlib.util.spec_from_file_location(pkg, folder / "__init__.py", submodule_search_locations=[str(folder)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = mod
        spec.loader.exec_module(mod)
        return importlib.import_module(f"{pkg}.build")
    spec = importlib.util.spec_from_file_location(f"{pkg}_build", folder / "build.py")  # EDGEQUADD: absolute imports
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def default_hp(name):
    return dict(load(name).DEFAULT_HP)
