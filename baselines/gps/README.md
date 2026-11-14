# GPS

## Source

Graph transformer built on PyTorch Geometric's `GPSConv` (Rampasek et al., NeurIPS 2022): GCN local message passing plus
multi-head global attention, with an RRWP positional encoding (random-walk landing probabilities, walk length 8;
Ma et al., ICML 2023).

## Graph

Directed edge i -> j wherever corr[i, j] > threshold (default 0.3), no self-loops, unweighted; the same graph as `gcn/`,
`graphsage/` and `gat/`. The RRWP encoding uses the same edge set, with correlation values as walk weights.

## Tunable settings

`channels` (64), `pe_dim` (32), `num_layers` (3), `dropout` (0.5, passed to `GPSConv`), `pooling` (mean), `threshold` (0.3),
`walk_length` (8), `heads` (4), `activation` (relu).

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model gps --seed 42
```
