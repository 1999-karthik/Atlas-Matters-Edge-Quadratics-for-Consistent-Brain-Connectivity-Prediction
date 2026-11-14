#!/bin/bash
# Reproduce the EDGEQUADD ABIDE results of the paper with the released tuned configs
# (configs/abide/edgequadd/abide_<atlas>.json).
#
#   bash scripts/run_abide.sh <atlas|all> [seeds] [out_dir]
#
#   atlas : AAL116 harvard48 schaefer100 | all
#   seeds : comma list, default 42,1042,2042,3042,4042
#   out   : default runs/abide_repro
#
# ABIDE data cannot be redistributed: obtain it from http://fcon_1000.projects.nitrc.org/indi/abide/ under its data use agreement and place
# the preprocessed files at Datasets/abide_data/abide_<atlas>.npy (override with DATA_DIR=...).
# Protocol: stratified 70/10/20 split per seed, checkpoint selected on validation loss, test evaluated once.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R"
PY=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-$R/Datasets}
ATLASES="AAL116 harvard48 kmeans100 schaefer100 ward100"
a_arg=${1:?atlas or all}
SEEDS=${2:-42,1042,2042,3042,4042}; OUT=${3:-$R/runs/abide_repro}
[ "$a_arg" = all ] && a_list=$ATLASES || a_list=$a_arg
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
for a in $a_list; do
  cfg=$R/configs/abide/edgequadd/abide_$a.json
  [ -f "$cfg" ] || { echo "no config: $cfg" >&2; exit 1; }
  for s in ${SEEDS//,/ }; do
    echo "== edgequadd abide_$a seed $s"
    $PY benchmark/run.py --dataset abide_$a --model edgequadd --seed $s --selection val_loss --strict_determinism \
        --data_dir "$DATA_DIR" --out "$OUT" --tuned "$cfg"
  done
done
echo "done -> $OUT"; echo "summarise: $PY benchmark/aggregate.py --runs $OUT"
