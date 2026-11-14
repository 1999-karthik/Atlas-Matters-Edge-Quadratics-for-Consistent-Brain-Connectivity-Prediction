# Baselines

Each baseline lives in its own folder with `model.py` (architecture), `build.py` (default hyper-parameters and the
constructor used by the runner) and a `README.md` describing its source and modifications. All models run through the
same runner, split, training loop and metrics:

```bash
python benchmark/run.py --dataset <dataset>_<atlas> --model <folder> --seed 42
```

| Folder | Method | Source | License |
|---|---|---|---|
| `gcn/`, `graphsage/`, `gat/` | GCN, GraphSAGE, GAT (tunedGNN recipe) | [tunedGNN](https://github.com/LUOyk1999/tunedGNN), NeurIPS 2024 | MIT |
| `gps/` | GPS graph transformer with RRWP encoding | PyTorch Geometric `GPSConv` | — |
| `gps/gcn_plain/` | Plain GCN (benchmark name `gcn_plain`) | this repository | — |
| `bnt/` | Brain Network Transformer | [BrainNetworkTransformer](https://github.com/Wayfear/BrainNetworkTransformer), NeurIPS 2022 | MIT |
| `brainnetcnn/` | BrainNetCNN | via BrainNetworkTransformer | MIT |
| `transformer/` | Graph transformer | via BrainNetworkTransformer | MIT |
| `alter/` | ALTER | [ALTER](https://github.com/yushuowiki/ALTER) | none published |
| `bqn/` | BQN | [BQN-demo](https://github.com/LYWJUN/BQN-demo) | none published |
| `biobgt/` | BioBGT | [BioBGT](https://github.com/pcyyyy/BioBGT), ICLR 2025 | none published |

License notices for the MIT-licensed sources are in `THIRD_PARTY_LICENSES/`.

**Credit.** Every baseline method and its original implementation are the work of their authors; please cite their
papers. The code in `alter/`, `bqn/` and `biobgt/` is adapted from repositories that publish no license: it is included
only to reproduce the comparisons in our paper, all rights remain with the authors, and we will remove any of these
folders at the authors' request.
