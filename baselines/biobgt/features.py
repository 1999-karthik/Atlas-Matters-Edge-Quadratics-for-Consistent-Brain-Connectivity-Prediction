"""Per-subject BioBGT inputs (Peng et al., "Biologically Plausible Brain Graph Transformer", ICLR 2025,
https://github.com/pcyyyy/BioBGT).

Two quantities are computed from each subject's own graph only (no labels, no other subjects, so they can be
cached once per data file and reused by every split/seed):

* VE        [R]      node importance = minimum vertex entanglement over the inverse temperature beta
                     (port of upstream `Model/node_perturbation.py::VertexEnt`; verified numerically).
* update_x  [R, 128] functional-module-aware node embeddings from the community contrastive extractor
                     (upstream `Model/FunctionalModuleExtractor.py`, repaired so that it trains).

Graph: Pearson correlation matrix, edges with corr <= threshold removed, diagonal removed. The paper uses threshold 0.3
for ABIDE and 0 for ADNI / ADHD-200 (`graph_threshold`); PPMI is not in the paper and uses 0.

Extractor (what is taken from where):
  paper    Louvain modules; edge dropping that removes inter-module edges first; GNN view encoder; InfoNCE where
           negatives are nodes of other modules (Eq. in Sec. 4.3).
  code     (the paper does not give these) 2-layer GCN T -> 256 -> 128 with PReLU, trained 100 steps with Adam lr 1e-3 per
           graph; python-louvain `best_partition`; edge keep-probability from community-strength weights (`ced`,
           drop rate p = 0.1); positive = the same node in the second view; negatives = nodes of other modules in the
           first view; InfoNCE temperature 0.1.
  repair   the released code wraps the embeddings in `torch.tensor(...)` copies before the loss (the encoder gets no
           gradient), feeds the 128-d output back into a conv that expects T inputs, and hard-codes 200 nodes. Here
           gradients flow, each step re-encodes the original features, and the node count comes from the data.
  option   variant="paper_neg" additionally uses the second view's nodes of other modules as negatives (the paper's
           Eq. has both views in the denominator). Default "code".
Louvain and edge dropping are seeded per subject, so the output is deterministic.
"""
import hashlib
import os
import sys
import multiprocessing as _mp
from pathlib import Path

import community  # python-louvain
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from torch_geometric.nn import GCNConv

VERSION = "biobgt-features-v3"  # v3: series are prepared with prepare_series (valid window, z-score, tile)
UPDATE_DIM = 128


def prepare_series(ts, orig_len=None, mode="tile"):
    """[S, R, T] time series -> [S, R, T] float32, per subject and without looking at any other subject or label.

    The dataset files store each series cropped or zero-padded to a fixed T (the number of real volumes is in
    `original_timeseries_lengths`). Zero padding would hand the model the scan length (in ADNI the 26 short scans are all one
    class) and the files are scaled differently (ADHD-200 is not z-scored). So for every subject: keep the valid window,
    z-score each ROI over it (dead ROIs stay 0) and repeat the window cyclically up to T, which leaves no zero tail.

    mode="crop": the repeat period of a short scan is still its length (a model can read it off the tiled series), so instead
    every subject is cut to its first T0 = (shortest valid window in the file) volumes and z-scored; the output is [S, R, T0],
    with no padding and no repetition. T0 uses scan lengths only, never labels.
    """
    ts = np.asarray(ts, dtype=np.float32)
    S, R, T = ts.shape
    if mode == "crop":
        lens = np.full(S, T) if orig_len is None else np.minimum(np.asarray(orig_len, dtype=int), T)
        T0 = int(lens.min())
        out = np.empty((S, R, T0), np.float32)
        for i in range(S):
            x = ts[i, :, :T0]
            sd = x.std(1, keepdims=True)
            out[i] = np.where(sd > 1e-8, (x - x.mean(1, keepdims=True)) / np.where(sd > 1e-8, sd, 1.0), 0.0)
        return out
    if mode != "tile":
        raise ValueError(f"unknown series mode {mode!r}; choose tile or crop")
    out = np.empty_like(ts)
    for i in range(S):
        L = T if orig_len is None else int(min(orig_len[i], T))
        x = ts[i, :, :L]
        sd = x.std(1, keepdims=True)
        z = np.where(sd > 1e-8, (x - x.mean(1, keepdims=True)) / np.where(sd > 1e-8, sd, 1.0), 0.0)
        out[i] = np.tile(z, (1, -(-T // L)))[:, :T]
    return out


def graph_threshold(dataset_family):
    """Edge threshold from the paper's Table 3: 0.3 for ABIDE, 0 for ADNI and ADHD-200 (PPMI: not in paper, 0)."""
    return 0.3 if dataset_family.lower() == "abide" else 0.0


def positive_graph(corr, threshold=0.0):
    A = np.asarray(corr, dtype=np.float64)
    A = np.where(A > threshold, A, 0.0)
    np.fill_diagonal(A, 0.0)
    return (A + A.T) / 2


def _betas(lambda2, gap=100):
    a = list(np.linspace(0, 5 / lambda2, gap))
    b = list(np.linspace(a[-1], 10 / lambda2, gap))
    return np.array(a[:-1] + b)


def _entropy(lam, betas):
    """Spectral entropy rows for each beta; lam is [N] (or [M,N] broadcast) -> [len(betas)] (or [len(betas), M])."""
    t = -betas[:, None] * lam[None, :]
    Z = np.exp(t).sum(1, keepdims=True)
    return -(np.exp(t) * (t / np.log(2) - np.log2(Z)) / Z).sum(1)


def vertex_entanglement(corr, threshold=0.0):
    A = positive_graph(corr, threshold)
    N = len(A)
    L = np.diag(A.sum(1)) - A
    eigval, eigvec = np.linalg.eigh(L)
    n_comp = connected_components(A > 0)[0]
    if n_comp >= N:  # no edge survives the threshold: upstream indexes past the spectrum; nothing to perturb, so VE = 0
        return np.zeros(N, dtype=np.float32)
    betas = _betas(eigval[n_comp])
    S = np.array([_entropy(eigval, np.array([b]))[0] for b in betas])
    lam_ral = np.tile(eigval, (N, 1))
    for v in range(N):
        nb = np.nonzero(A[v] > 0)[0]
        kx = len(nb)
        if kx == 0:
            continue
        A_loc = A[np.ix_(nb, nb)]
        N_loc = kx + A_loc.sum() / 2
        weight = 2 * N_loc / kx / (kx + 1)
        if weight == 1:
            continue
        nb_all = np.sort(np.append(nb, v))
        dA = weight - A[np.ix_(nb_all, nb_all)]
        dA = dA - np.diag([weight] * (kx + 1))
        dL = np.diag(dA.sum(1)) - dA
        V = eigvec[nb_all, :]                     # [k+1, N]
        lam_ral[v] = eigval + ((dL @ V) * V).sum(0)
    E = np.empty((len(betas), N))
    for x in range(N):
        t = -betas[:, None] * lam_ral[x][None, :]
        Z = np.exp(t).sum(1, keepdims=True)
        E[:, x] = -(np.exp(t) * (t / np.log(2) - np.log2(Z)) / Z).sum(1) - S
    return E.min(0).astype(np.float32)


class _Encoder(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1, self.c2 = GCNConv(in_ch, 2 * out_ch), GCNConv(2 * out_ch, out_ch)
        self.act = torch.nn.PReLU()

    def forward(self, x, ei):
        return self.act(self.c2(self.act(self.c1(x, ei)), ei))


def _info_nce(q, p, neg, temperature=0.1):
    q, p, neg = F.normalize(q, dim=-1), F.normalize(p, dim=-1), F.normalize(neg, dim=-1)
    logits = torch.cat([(q * p).sum(-1, keepdim=True), q @ neg.T], dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long))


def _edge_weights(edge_index, comm, node_cs):
    if edge_index.shape[1] == 0:  # edgeless graph (no edge above the threshold): nothing to weight or drop
        return torch.zeros(0)
    cu, cv = comm[edge_index[0]], comm[edge_index[1]]
    w = np.where(cu == cv, node_cs[edge_index[0]], -(node_cs[edge_index[0]] + node_cs[edge_index[1]]))
    rng = w.max() - w.min()
    return torch.from_numpy((w - w.min()) / rng if rng > 0 else np.ones_like(w)).float()


def module_features(ts, corr, seed, threshold=0.0, epochs=100, variant="code"):
    """ts [R,T] raw time series, corr [R,R] -> [R, 128] float32."""
    A = positive_graph(corr, threshold)
    N = len(A)
    G = nx.from_numpy_array(A)
    part = community.best_partition(G, random_state=seed)      # python-louvain, as upstream
    groups = {}
    for node, c in part.items():
        groups.setdefault(c, []).append(node)
    comms = [sorted(v) for _, v in sorted(groups.items())]
    comm = np.zeros(N, dtype=int)
    for k, c in enumerate(comms):
        comm[c] = k
    links = A.sum() / 2
    deg = A.sum(1)
    node_cs = np.zeros(N, dtype=np.float32)
    for k, c in enumerate(comms):
        if links == 0:  # edgeless graph: community strength undefined, left at 0
            break
        inc = A[np.ix_(c, c)].sum() / 2
        node_cs[c] = inc / links - (deg[c].sum() / (2 * links)) ** 2
    src, dst = np.nonzero(A > 0)
    edge_index = torch.from_numpy(np.stack([src, dst])).long()

    g = torch.Generator().manual_seed(seed)
    w = _edge_weights(edge_index.numpy(), comm, node_cs)
    w = (w / w.mean() * 0.9).clamp(max=1.0) if len(w) else w       # p = 0.1, threshold = 1.0 as upstream `ced`
    keep = torch.bernoulli(w, generator=g).bool()
    edge_aug = edge_index[:, keep]

    torch.manual_seed(seed)
    x = torch.from_numpy(np.asarray(ts, dtype=np.float32))
    enc = _Encoder(x.shape[1], UPDATE_DIM)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    for _ in range(epochs):
        h, h_aug = enc(x, edge_index), enc(x, edge_aug)
        loss = 0.0
        for c in comms:
            others = np.setdiff1d(np.arange(N), c)
            if len(others) == 0:
                continue
            neg = h[others] if variant == "code" else torch.cat([h[others], h_aug[others]])
            loss = loss + _info_nce(h[c], h_aug[c], neg)
        if not torch.is_tensor(loss):
            break
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return enc(x, edge_index).numpy().astype(np.float32)


def _importable_self():
    """Import this module as `biobgt.features` so that multiprocessing workers can unpickle its functions.

    The benchmark loads baselines under dynamic package names that are not on sys.path; adding baselines/ to
    sys.path and importing by a stable name lets spawned workers resolve the worker function.
    """
    import importlib
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("biobgt.features")


def _one(args):
    torch.set_num_threads(1)
    i, ts, corr, threshold, variant = args
    return i, vertex_entanglement(corr, threshold), module_features(ts, corr, seed=1000003 + i, threshold=threshold,
                                                                    variant=variant)


def compute(ts_srt, corr, threshold, variant="code", workers=8):
    """ts_srt [S,R,T] raw time series, corr [S,R,R] -> (VE [S,R], update_x [S,R,128])."""
    S = len(corr)
    ve = np.zeros((S, corr.shape[1]), np.float32)
    ux = np.zeros((S, corr.shape[1], UPDATE_DIM), np.float32)
    jobs = [(i, ts_srt[i], corr[i], threshold, variant) for i in range(S)]
    worker = _importable_self()._one
    with _mp.get_context('spawn').Pool(workers) as pool:
        for n, (i, v, u) in enumerate(pool.imap_unordered(worker, jobs, chunksize=1)):
            ve[i], ux[i] = v, u
            if n % 50 == 0:
                print(f"  biobgt features {n}/{S}", file=sys.stderr, flush=True)
    return ve, ux


def file_key(path, threshold, variant):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return f"{h.hexdigest()[:16]}-{VERSION}-thr{threshold}-{variant}"


def load_or_compute(npy_path, ts_srt, corr, cache_dir, threshold, variant="code", workers=8):
    """Cache <cache_dir>/<stem>.npz keyed by the source file hash + VERSION; returns (VE, update_x) for ALL subjects."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / (Path(npy_path).stem + ".npz")
    key = file_key(npy_path, threshold, variant)
    if out.exists():
        z = np.load(out, allow_pickle=False)
        if str(z["key"]) == key:
            return z["ve"], z["ux"]
    ve, ux = compute(ts_srt, corr, threshold, variant, workers)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.npz")   # atomic: parallel jobs may race on the same file
    np.savez(tmp, ve=ve, ux=ux, key=np.array(key))
    os.replace(tmp, out)
    return ve, ux
