# ADNI configs (paper results)

Tuned EDGEQUADD hyper-parameters for every ADNI atlas: `edgequadd/adni_<atlas>.json`
(Optuna best trial, selected on validation loss; test data never used for tuning).

## Data

ADNI data cannot be redistributed. Request access from the provider under its data use agreement:
https://adni.loni.usc.edu. The configs and scripts here expect the preprocessed
connectivity files at `Datasets/adni_data/adni_<atlas>.npy` (override with `DATA_DIR=`).

## Run

```bash
# one atlas, 5 seeds
bash scripts/run_adni.sh harvard48

# all 5 atlases
bash scripts/run_adni.sh all

# single run by hand
python benchmark/run.py --dataset adni_schaefer100 --model edgequadd --seed 42 \
    --selection val_loss --strict_determinism --tuned configs/adni/edgequadd/adni_schaefer100.json

# summarise (mean ± sample std over seeds)
python benchmark/aggregate.py --runs runs/adni_repro
```

Seeds: 42, 1042, 2042, 3042, 4042. Each seed fixes a stratified 70/10/20 split and the weight initialisation;
the checkpoint is chosen on validation loss.

## Expected EDGEQUADD test results (%, mean ± sample std over 5 seeds; NVIDIA A30, torch 2.9.1+cu128)

| Atlas | AUC | ACC | SEN | SPE |
|---|---|---|---|---|
| AAL116 | 85.10 ± 9.66 | 77.93 ± 7.94 | 71.67 ± 29.23 | 82.35 ± 16.11 |
| harvard48 | 85.39 ± 1.94 | 75.17 ± 8.93 | 66.67 ± 21.25 | 81.18 ± 23.31 |
| kmeans100 | 44.80 ± 8.60 | 53.10 ± 8.66 | 16.67 ± 17.68 | 78.82 ± 23.01 |
| schaefer100 | 90.69 ± 5.13 | 77.24 ± 10.79 | 75.00 ± 21.25 | 78.82 ± 18.88 |
| ward100 | 50.59 ± 9.14 | 53.10 ± 7.94 | 45.00 ± 36.13 | 58.82 ± 23.89 |

### Per seed (test, %)

| Atlas | Seed | AUC | ACC | SEN | SPE | best epoch |
|---|---|---|---|---|---|---|
| AAL116 | 42 | 93.63 | 75.86 | 100.00 | 58.82 | 22 |
| AAL116 | 1042 | 90.20 | 86.21 | 75.00 | 94.12 | 2 |
| AAL116 | 2042 | 79.90 | 68.97 | 25.00 | 100.00 | 5 |
| AAL116 | 3042 | 91.18 | 86.21 | 91.67 | 82.35 | 6 |
| AAL116 | 4042 | 70.59 | 72.41 | 66.67 | 76.47 | 27 |
| harvard48 | 42 | 82.84 | 62.07 | 91.67 | 41.18 | 4 |
| harvard48 | 1042 | 87.75 | 72.41 | 33.33 | 100.00 | 9 |
| harvard48 | 2042 | 85.29 | 79.31 | 66.67 | 88.24 | 151 |
| harvard48 | 3042 | 86.76 | 86.21 | 75.00 | 94.12 | 3 |
| harvard48 | 4042 | 84.31 | 75.86 | 66.67 | 82.35 | 35 |
| kmeans100 | 42 | 43.63 | 37.93 | 25.00 | 47.06 | 57 |
| kmeans100 | 1042 | 33.33 | 58.62 | 0.00 | 100.00 | 0 |
| kmeans100 | 2042 | 50.00 | 58.62 | 0.00 | 100.00 | 0 |
| kmeans100 | 3042 | 41.18 | 55.17 | 16.67 | 82.35 | 0 |
| kmeans100 | 4042 | 55.88 | 55.17 | 41.67 | 64.71 | 4 |
| schaefer100 | 42 | 92.65 | 79.31 | 83.33 | 76.47 | 272 |
| schaefer100 | 1042 | 86.27 | 65.52 | 91.67 | 47.06 | 7 |
| schaefer100 | 2042 | 89.71 | 79.31 | 66.67 | 88.24 | 119 |
| schaefer100 | 3042 | 98.53 | 93.10 | 91.67 | 94.12 | 13 |
| schaefer100 | 4042 | 86.27 | 68.97 | 41.67 | 88.24 | 238 |
| ward100 | 42 | 44.61 | 48.28 | 50.00 | 47.06 | 13 |
| ward100 | 1042 | 66.67 | 55.17 | 100.00 | 23.53 | 13 |
| ward100 | 2042 | 47.06 | 44.83 | 0.00 | 76.47 | 159 |
| ward100 | 3042 | 49.02 | 65.52 | 41.67 | 82.35 | 3 |
| ward100 | 4042 | 45.59 | 51.72 | 33.33 | 64.71 | 5 |

Machine-readable: `summary.csv` (mean ± std) and `per_seed.csv`.

Small differences (≈1 AUC point) can appear on other GPUs or library versions.
