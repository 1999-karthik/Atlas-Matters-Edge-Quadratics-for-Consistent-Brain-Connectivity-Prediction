# PPMI configs

Tuned EDGEQUADD hyper-parameters for PPMI (4 classes): `edgequadd/ppmi_<atlas>.json`
(selected on validation AUC; test data never used for tuning).

## Data

PPMI data cannot be redistributed. Obtain it from https://www.ppmi-info.org under its data use agreement and place the
preprocessed connectivity files at `Datasets/ppmi_data/ppmi_<atlas>.npy` (override with `DATA_DIR=`).

## Run

```bash
# one atlas, 5 seeds
bash scripts/run_ppmi.sh harvard48

# all released atlases
bash scripts/run_ppmi.sh all

# single run by hand
python benchmark/run.py --dataset ppmi_harvard48 --model edgequadd --seed 42 \
    --selection val_auc --strict_determinism --tuned configs/ppmi/edgequadd/ppmi_harvard48.json

# summarise (mean ± sample std over seeds)
python benchmark/aggregate.py --runs runs/ppmi_repro
```

Seeds: 42, 1042, 2042, 3042, 4042. Each seed fixes a stratified 70/10/20 split and the weight initialisation;
the checkpoint is chosen on validation AUC. All runs use `--strict_determinism`.

## Expected EDGEQUADD test results (%, mean ± sample std over 5 seeds; torch 2.9.1+cu128)

AUC is macro one-vs-rest; SEN and SPE are macro averages over the four classes (SEN equals balanced accuracy).

| Atlas | AUC | ACC | SEN | SPE |
|---|---|---|---|---|
| AAL116 | 56.26 ± 8.82 | 55.00 ± 3.06 | 28.21 ± 4.10 | 76.65 ± 2.18 |
| harvard48 | 53.20 ± 4.38 | 41.50 ± 8.22 | 23.74 ± 3.51 | 74.20 ± 2.62 |
| kmeans100 | 53.14 ± 6.83 | 47.50 ± 10.00 | 23.97 ± 4.33 | 75.46 ± 3.70 |
| schaefer100 | 57.01 ± 9.83 | 52.50 ± 10.16 | 26.88 ± 7.78 | 76.12 ± 2.77 |
| ward100 | 51.74 ± 7.08 | 53.00 ± 5.70 | 25.65 ± 2.52 | 76.02 ± 1.76 |

### Per seed (test, %)

| Atlas | Seed | AUC | ACC | SEN | SPE | best epoch |
|---|---|---|---|---|---|---|
| AAL116 | 42 | 63.61 | 50.00 | 33.73 | 76.67 | 218 |
| AAL116 | 1042 | 63.26 | 57.50 | 30.93 | 79.87 | 71 |
| AAL116 | 2042 | 42.42 | 55.00 | 23.91 | 74.32 | 88 |
| AAL116 | 3042 | 53.13 | 57.50 | 25.00 | 75.00 | 20 |
| AAL116 | 4042 | 58.89 | 55.00 | 27.47 | 77.37 | 187 |
| harvard48 | 42 | 52.20 | 50.00 | 28.85 | 77.98 | 71 |
| harvard48 | 1042 | 58.94 | 40.00 | 23.32 | 74.44 | 17 |
| harvard48 | 2042 | 56.36 | 50.00 | 25.30 | 74.85 | 131 |
| harvard48 | 3042 | 50.13 | 32.50 | 21.25 | 72.84 | 142 |
| harvard48 | 4042 | 48.38 | 35.00 | 19.96 | 70.89 | 34 |
| kmeans100 | 42 | 60.10 | 52.50 | 25.20 | 77.49 | 46 |
| kmeans100 | 1042 | 59.72 | 62.50 | 30.73 | 80.75 | 11 |
| kmeans100 | 2042 | 53.12 | 37.50 | 21.05 | 73.58 | 2 |
| kmeans100 | 3042 | 45.09 | 42.50 | 19.66 | 71.23 | 78 |
| kmeans100 | 4042 | 47.65 | 42.50 | 23.22 | 74.27 | 78 |
| schaefer100 | 42 | 56.24 | 37.50 | 19.86 | 73.48 | 72 |
| schaefer100 | 1042 | 72.48 | 52.50 | 24.01 | 75.66 | 11 |
| schaefer100 | 2042 | 56.15 | 50.00 | 25.30 | 75.64 | 64 |
| schaefer100 | 3042 | 55.11 | 65.00 | 40.25 | 80.82 | 79 |
| schaefer100 | 4042 | 45.09 | 57.50 | 25.00 | 75.00 | 4 |
| ward100 | 42 | 51.17 | 47.50 | 24.21 | 74.17 | 72 |
| ward100 | 1042 | 40.08 | 57.50 | 26.19 | 76.22 | 89 |
| ward100 | 2042 | 58.21 | 52.50 | 25.20 | 75.71 | 18 |
| ward100 | 3042 | 56.29 | 47.50 | 23.02 | 75.15 | 146 |
| ward100 | 4042 | 52.97 | 60.00 | 29.64 | 78.86 | 17 |

Machine-readable: `summary.csv` (mean ± std) and `per_seed.csv`.
Small differences can appear on other GPUs or library versions; the GPU used for each atlas is listed in `summary.csv`.
