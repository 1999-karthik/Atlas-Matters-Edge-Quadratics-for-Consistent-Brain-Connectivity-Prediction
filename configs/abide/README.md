# ABIDE configs

Tuned EDGEQUADD hyper-parameters for ABIDE: `edgequadd/abide_<atlas>.json`
(selected on validation loss; test data never used for tuning).

## Data

ABIDE data cannot be redistributed. Obtain it from http://fcon_1000.projects.nitrc.org/indi/abide/ and place the preprocessed
connectivity files at `Datasets/abide_data/abide_<atlas>.npy` (override with `DATA_DIR=`).

## Run

```bash
# one atlas, 5 seeds
bash scripts/run_abide.sh schaefer100

# all released atlases
bash scripts/run_abide.sh all

# single run by hand
python benchmark/run.py --dataset abide_schaefer100 --model edgequadd --seed 42 \
    --selection val_loss --strict_determinism --tuned configs/abide/edgequadd/abide_schaefer100.json

# summarise (mean ± sample std over seeds)
python benchmark/aggregate.py --runs runs/abide_repro
```

Seeds: 42, 1042, 2042, 3042, 4042. Each seed fixes a stratified 70/10/20 split and the weight initialisation;
the checkpoint is chosen on validation loss.

## Expected EDGEQUADD test results (%, mean ± sample std over 5 seeds; torch 2.9.1+cu128)

| Atlas | AUC | ACC | SEN | SPE |
|---|---|---|---|---|
| AAL116 | 70.25 ± 2.07 | 64.76 ± 3.25 | 61.02 ± 4.64 | 68.15 ± 4.66 |
| harvard48 | 68.80 ± 4.28 | 63.01 ± 3.93 | 58.37 ± 12.61 | 67.22 ± 8.99 |
| kmeans100 | 47.67 ± 4.79 | 50.20 ± 4.73 | 5.71 ± 12.78 | 90.55 ± 20.61 |
| schaefer100 | 71.57 ± 4.04 | 65.83 ± 3.58 | 62.86 ± 6.40 | 68.52 ± 3.14 |
| ward100 | 50.59 ± 2.34 | 51.26 ± 2.88 | 18.57 ± 26.00 | 80.93 ± 27.79 |

### Per seed (test, %)

| Atlas | Seed | AUC | ACC | SEN | SPE | best epoch |
|---|---|---|---|---|---|---|
| AAL116 | 42 | 71.26 | 67.48 | 59.18 | 75.00 | 21 |
| AAL116 | 1042 | 71.45 | 68.45 | 66.33 | 70.37 | 23 |
| AAL116 | 2042 | 67.23 | 60.68 | 54.08 | 66.67 | 22 |
| AAL116 | 3042 | 72.27 | 64.56 | 63.27 | 65.74 | 19 |
| AAL116 | 4042 | 69.04 | 62.62 | 62.24 | 62.96 | 31 |
| harvard48 | 42 | 69.12 | 66.50 | 58.16 | 74.07 | 13 |
| harvard48 | 1042 | 65.59 | 60.68 | 63.27 | 58.33 | 4 |
| harvard48 | 2042 | 63.44 | 57.28 | 38.78 | 74.07 | 15 |
| harvard48 | 3042 | 72.63 | 64.56 | 73.47 | 56.48 | 15 |
| harvard48 | 4042 | 73.22 | 66.02 | 58.16 | 73.15 | 12 |
| kmeans100 | 42 | 48.31 | 51.94 | 0.00 | 99.07 | 4 |
| kmeans100 | 1042 | 43.02 | 41.75 | 28.57 | 53.70 | 5 |
| kmeans100 | 2042 | 42.58 | 52.43 | 0.00 | 100.00 | 13 |
| kmeans100 | 3042 | 53.36 | 52.43 | 0.00 | 100.00 | 0 |
| kmeans100 | 4042 | 51.07 | 52.43 | 0.00 | 100.00 | 0 |
| schaefer100 | 42 | 73.69 | 68.93 | 68.37 | 69.44 | 31 |
| schaefer100 | 1042 | 71.83 | 67.48 | 69.39 | 65.74 | 27 |
| schaefer100 | 2042 | 64.98 | 59.71 | 54.08 | 64.81 | 29 |
| schaefer100 | 3042 | 71.61 | 66.99 | 63.27 | 70.37 | 30 |
| schaefer100 | 4042 | 75.73 | 66.02 | 59.18 | 72.22 | 35 |
| ward100 | 42 | 47.07 | 46.12 | 54.08 | 38.89 | 45 |
| ward100 | 1042 | 50.34 | 52.43 | 0.00 | 100.00 | 11 |
| ward100 | 2042 | 53.12 | 52.43 | 0.00 | 100.00 | 6 |
| ward100 | 3042 | 52.29 | 52.91 | 38.78 | 65.74 | 20 |
| ward100 | 4042 | 50.12 | 52.43 | 0.00 | 100.00 | 6 |

Machine-readable: `summary.csv` (mean ± std) and `per_seed.csv`.
Small differences can appear on other GPUs or library versions; the GPU used for each atlas is listed in `summary.csv`.
