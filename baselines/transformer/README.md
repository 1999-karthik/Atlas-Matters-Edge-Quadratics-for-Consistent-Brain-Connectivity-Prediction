# Graph transformer

## Source

A stack of `TransformerEncoderLayer` blocks over the correlation rows with mean readout, from the Brain Network Transformer
code base (https://github.com/Wayfear/BrainNetworkTransformer, MIT, `models/transformer.py`).

## Modifications

- Number of attention heads: the largest divisor of the feature size that is at most 4.
- Output width follows the number of classes.

## Tunable settings

`dropout` (default 0.1), `ffn_dim` (default 1024) and `heads` (default 4); the defaults reproduce the upstream network.

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model transformer --seed 42
```
