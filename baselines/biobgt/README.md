# BioBGT

> **Credit.** This method and its original implementation are the work of its authors (https://github.com/pcyyyy/BioBGT). This folder is an adapted copy included only to reproduce the comparison in our paper. The upstream repository publishes no license, so all rights remain with the authors; please cite their paper. We will remove this folder at the authors' request.

## Source

Peng et al., *Biologically Plausible Brain Graph Transformer*, ICLR 2025 (https://arxiv.org/abs/2502.08958),
code at https://github.com/pcyyyy/BioBGT. Complete self-attention whose queries and keys come from functional-module-aware
node embeddings (`update_x`), combined with a network-entanglement node-importance encoding (`VE`).

## Modifications

- `model.py` is a dense, batched form of upstream `Model/models.py` and `Model/layers.py`, including per-subject
  BatchNorm during training; its outputs match the upstream model to about 1e-7.
- `features.py` ports the vertex-entanglement computation and provides a functional-module extractor that trains end to
  end, which the released extractor does not do as-is.
- Defaults follow the paper's Table 3 and the authors' training script: AdamW 3e-4, 10 warm-up epochs, 3 layers, 8 heads,
  hidden size 128, dropout 0.5 on ABIDE and 0.1 elsewhere, edge threshold 0.3 on ABIDE and 0 elsewhere.
- Mixup is disabled because the model takes three input tensors.

## Per-subject inputs

`VE` and `update_x` are computed from each subject's own graph only (no labels, no other subjects) and cached per data
file under `<out>/_cache/biobgt/`. To compute them ahead of the runs:

```bash
python benchmark/precompute_biobgt.py --data_dir Datasets --out runs
```

## Tunable settings

`d_model` (128), `num_heads` (8), `num_layers` (3), `dropout`, `ffn_mult` (2) and `activation` (relu). `update_x` stays
128-wide; the query/key projection maps it to `2 * d_model`, which equals the upstream layer at `d_model = 128`.

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model biobgt --seed 42
```
