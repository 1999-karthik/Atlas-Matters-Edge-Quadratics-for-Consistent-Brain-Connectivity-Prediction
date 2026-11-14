# GCN (plain)

## Source

Baseline graph convolutional network: input projection, three `GCNConv` layers with BatchNorm, ReLU and dropout, mean
pooling and an MLP head. Node features are the rows of the correlation matrix.

## Graph

Directed edge i -> j wherever corr[i, j] > threshold (default 0.3), no self-loops, unweighted; the same graph as `gcn/`,
`graphsage/` and `gat/`.

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model gcn_plain --seed 42
```
