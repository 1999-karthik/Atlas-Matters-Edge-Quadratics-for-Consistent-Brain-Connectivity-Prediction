# BQN

> **Credit.** This method and its original implementation are the work of its authors (https://github.com/LYWJUN/BQN-demo). This folder is an adapted copy included only to reproduce the comparison in our paper. The upstream repository publishes no license, so all rights remain with the authors; please cite their paper. We will remove this folder at the authors' request.

## Source

https://github.com/LYWJUN/BQN-demo (`model/BQN.py`, `model/cluster_pooling.py`). Node-wise quadratic perceptron layers
(`MLP_R(corr) * feature + MLP_G(feature^2)`) followed by an orthogonal-clustering readout whose encoder acts on the
flattened graph.

## Modifications

- Relative imports instead of the `BQN_Demo.model` package path.
- Output width follows the number of classes.
- `pooling=False` raises a `ValueError`, since the classifier exists only with pooling.

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model bqn --seed 42
```
