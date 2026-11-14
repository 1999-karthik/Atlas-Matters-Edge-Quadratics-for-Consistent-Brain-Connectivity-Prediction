# ALTER

> **Credit.** This method and its original implementation are the work of its authors (https://github.com/yushuowiki/ALTER). This folder is an adapted copy included only to reproduce the comparison in our paper. The upstream repository publishes no license, so all rights remain with the authors; please cite their paper. We will remove this folder at the authors' request.

## Source

Brain graph transformer with long-range random-walk positional encoding. Adapted from `alter/models/LRBGT` in
https://github.com/yushuowiki/ALTER: the BNT backbone with an RRWP positional encoding (walk length and embedding size 32).
Every positive edge enters the random walk, as in the public release. The biased-random-walk strategy described in the
paper is not part of the public code and is not included.

## Modifications

- Device placement follows the input (no hard-coded `.cuda()`).
- Number of attention heads: the largest divisor of the feature size that is at most 4.
- Output width follows the number of classes.

## Tunable settings

`dropout` (default 0.1) and `ffn_dim` (default 1024); the defaults reproduce the upstream network.

## Run

```bash
python benchmark/run.py --dataset abide_AAL116 --model alter --seed 42
```
