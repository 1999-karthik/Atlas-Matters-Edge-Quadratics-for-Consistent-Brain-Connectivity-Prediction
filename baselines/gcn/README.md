# GCN (tunedGNN recipe)

## Source

Luo et al., *Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification*, NeurIPS 2024
(https://github.com/LUOyk1999/tunedGNN, MIT). The `MPNNs` class re-expresses the upstream `medium_graph/model.py`:
residual connections, LayerNorm and dropout around standard message-passing layers. Layer: `GCNConv`.

## Modifications

- Correlation matrix to graph: node features are correlation rows; edge i -> j wherever corr[i, j] > threshold
  (default 0.3), no self-loops, unweighted.
- Graph-level readout: mean pooling followed by an MLP.
- Upstream hyper-parameters are tuned for node classification; here hidden size, depth and dropout are tuned on the
  validation set.

## Tunable settings

`hidden` (64), `layers` (3), `dropout` (0.5), `res`, `ln`, `bn`, `jk`, `threshold` (0.3), `activation` (relu).

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model gcn --seed 42
```
