# Brain Network Transformer (BNT)

## Source

Kan et al., *Brain Network Transformer*, NeurIPS 2022 (https://github.com/Wayfear/BrainNetworkTransformer, MIT),
`source/models/BNT`. Transformer encoder layers with orthogonal-clustering (OCRead) pooling; default `pos_encoding: none`.

## Modifications

- Number of attention heads: the largest divisor of the feature size that is at most 4.
- Output width follows the number of classes.
- The transformer encoder accepts the `is_causal` argument required by PyTorch >= 2.

## Tunable settings

`dropout` (default 0.1) and `ffn_dim` (default 1024); the defaults reproduce the upstream network.

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model bnt --seed 42
```
