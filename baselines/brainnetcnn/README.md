# BrainNetCNN

## Source

Kawahara et al., *BrainNetCNN*, NeuroImage 2017, as implemented in the Brain Network Transformer code base
(https://github.com/Wayfear/BrainNetworkTransformer, MIT). Edge-to-edge, edge-to-node and node-to-graph convolutions on the
correlation matrix.

## Modifications

- Output width follows the number of classes.
- Dropout is applied only in training mode (`F.dropout(..., training=self.training)`).

## Tunable settings

`dropout` (default 0.5) and `width_mult` (default 1.0); the defaults reproduce the upstream network.

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model brainnetcnn --seed 42
```
