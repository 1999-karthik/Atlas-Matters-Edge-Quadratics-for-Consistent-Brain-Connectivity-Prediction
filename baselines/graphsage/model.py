"""Classic GNN baselines (GCN, GraphSAGE, GAT) following "Classic GNNs are Strong Baselines: Reassessing GNNs for
Node Classification" (Luo et al., NeurIPS 2024), https://github.com/LUOyk1999/tunedGNN (MIT License).

`MPNNs` below re-expresses that repository's `medium_graph/model.py` (same layers, residual `lins`, LayerNorm/BatchNorm,
dropout, optional JK). Its encoder output is numerically identical to upstream (max abs diff 0.0 for GCN/SAGE/GAT, all
res/ln/bn/jk/pre_linear settings, see scripts/check_baseline_fidelity.py); the only differences are that GAT with several
heads uses per-head width hidden/heads (so layer widths stay consistent) and the final node-level `pred_local` linear is
replaced by `forward_features` + the graph readout below.
`ClassicGNN` adds what the original lacks because it targets node classification: a graph builder for dense
correlation matrices and a mean-pool + MLP graph readout.

Graph: node i has feature vector = row i of the correlation matrix; undirected edge (i, j) when corr[i, j] > threshold
(default 0.3, the same graph used for the GCN/GPS baselines), no self loops (GCN adds its own).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}


def make_activation(name):
    """Activation module by name; raises ValueError for an unknown name."""
    try:
        return ACTIVATIONS[str(name).lower()]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}") from None


class MPNNs(nn.Module):
    def __init__(self, in_channels, hidden_channels, local_layers=3, dropout=0.5, heads=1, pre_ln=False,
                 pre_linear=False, res=False, ln=False, bn=False, jk=False, gnn='gcn', activation='relu'):
        super().__init__()
        self.act = make_activation(activation)
        assert gnn in ('gcn', 'sage', 'gat')
        if gnn == 'gat':
            assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads"
        self.dropout, self.pre_ln, self.pre_linear = dropout, pre_ln, pre_linear
        self.res, self.ln, self.bn, self.jk = res, ln, bn, jk

        self.local_convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()
        if pre_ln:
            self.pre_lns = nn.ModuleList()
        if pre_linear:
            self.lin_in = nn.Linear(in_channels, hidden_channels)

        def conv(i, o):
            if gnn == 'gat':
                return GATConv(i, o // heads, heads=heads, concat=True, add_self_loops=False, bias=False)
            if gnn == 'sage':
                return SAGEConv(i, o)
            return GCNConv(i, o, cached=False, normalize=True)

        n_layers = local_layers
        first_in = in_channels
        if pre_linear:
            first_in = hidden_channels
        for k in range(n_layers):
            cin = first_in if k == 0 else hidden_channels
            self.local_convs.append(conv(cin, hidden_channels))
            # only what forward_features uses is created, so n_params counts trainable parameters and nothing idles
            # (upstream builds all of them regardless; the state-dict keys that exist here are a subset of upstream's)
            if res:
                self.lins.append(nn.Linear(cin, hidden_channels))
            if ln:
                self.lns.append(nn.LayerNorm(hidden_channels))
            elif bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if pre_ln:
                self.pre_lns.append(nn.LayerNorm(cin))

    def forward_features(self, x, edge_index):
        if self.pre_linear:
            x = F.dropout(self.lin_in(x), p=self.dropout, training=self.training)
        x_final = 0
        for i, local_conv in enumerate(self.local_convs):
            x = local_conv(x, edge_index) + self.lins[i](x) if self.res else local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_final = x_final + x if self.jk else x
        return x_final


def dense_to_batch_graph(corr, threshold):
    """corr [B,N,N] -> (x [B*N, N], edge_index [2, E]) with edges where corr > threshold (diagonal removed)."""
    B, N, _ = corr.shape
    mask = corr > threshold
    mask = mask & ~torch.eye(N, dtype=torch.bool, device=corr.device).unsqueeze(0)
    b, i, j = torch.nonzero(mask, as_tuple=True)
    off = b * N
    return corr.reshape(B * N, N), torch.stack([i + off, j + off])


class ClassicGNN(nn.Module):
    def __init__(self, n_roi, num_classes, gnn='sage', hidden=64, layers=3, dropout=0.5, heads=1, res=True, ln=True,
                 bn=False, jk=False, pre_linear=False, threshold=0.3, activation='relu'):
        super().__init__()
        self.threshold = threshold
        self.encoder = MPNNs(n_roi, hidden, local_layers=layers, dropout=dropout, heads=heads, res=res, ln=ln,
                             bn=bn, jk=jk, pre_linear=pre_linear, gnn=gnn, activation=activation)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, num_classes))

    def forward(self, corr):
        B, N, _ = corr.shape
        x, edge_index = dense_to_batch_graph(corr, self.threshold)
        h = self.encoder.forward_features(x, edge_index).view(B, N, -1).mean(dim=1)
        return self.head(h)
