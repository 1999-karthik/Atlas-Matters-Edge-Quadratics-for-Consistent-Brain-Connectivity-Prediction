"""Dense, batched BioBGT graph transformer (Peng et al., ICLR 2025; https://github.com/pcyyyy/BioBGT).

Equivalent to upstream `Model/models.py::GraphTransformer` + `Model/layers.py` run on one subject at a time, but
batched: complete self-attention in which queries/keys come from the functional-module embedding `update_x`
(shared by every layer) and values from the running node representation; node input = Linear(T -> d)(time series)
+ Linear(1 -> d)(vertex entanglement); post-norm layers with BatchNorm; per-node MLP classifier, logits averaged
over nodes. BatchNorm is applied per subject exactly as upstream does when it feeds subjects one at a time
(statistics over that subject's nodes in training, running statistics updated subject by subject, running statistics at
evaluation), see `_GraphBN`; numerically checked against upstream in scripts/check_baseline_fidelity.py.
"""
import torch
import torch.nn as nn


class BioBGTLayer(nn.Module):
    def __init__(self, d, heads, dim_ff, dropout, batch_norm=True, activation="relu", update_dim=None):
        super().__init__()
        self.act = {"relu": torch.relu, "gelu": torch.nn.functional.gelu, "elu": torch.nn.functional.elu,
                    "leaky_relu": torch.nn.functional.leaky_relu}[activation]
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        # queries / keys come from the module-aware embedding update_x, whose width (update_dim, 128 from the extractor) is independent
        # of the hidden size d; upstream has update_dim == d, which is the default here
        self.to_qk = nn.Linear(d if update_dim is None else update_dim, 2 * d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)
        self.lin1, self.lin2 = nn.Linear(d, dim_ff), nn.Linear(dim_ff, d)
        self.drop, self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        mk = (lambda: _GraphBN(d)) if batch_norm else (lambda: nn.LayerNorm(d))
        self.norm1, self.norm2 = mk(), mk()
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

    def forward(self, x, update_x):
        B, N, d = x.shape
        a0, a1 = self.to_qk(update_x).chunk(2, dim=-1)           # upstream: qk = (a0, a1); score[i, j] = a1_i . a0_j
        split = lambda t: t.view(B, N, self.h, self.dh).transpose(1, 2)  # noqa: E731
        scores = split(a1) @ split(a0).transpose(-1, -2) * self.dh ** -0.5
        attn = self.attn_drop(scores.softmax(-1))
        out = (attn @ split(self.to_v(x))).transpose(1, 2).reshape(B, N, d)
        x = self.norm1(x + self.drop1(self.out_proj(out)))
        x = self.norm2(x + self.drop2(self.lin2(self.drop(self.act(self.lin1(x))))))
        return x


class _GraphBN(nn.Module):
    """BatchNorm1d over the nodes of ONE graph at a time, for a batch of graphs.

    Upstream runs the model on one subject per call, so in training BatchNorm1d normalises with that subject's own
    node statistics and updates its running statistics once per subject; at evaluation it uses the running
    statistics. This reproduces that for a batch [B, N, d] (the running-stat update is the same B sequential updates).
    """

    def __init__(self, d, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight, self.bias = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            mean = x.mean(1)
            var = x.var(1, unbiased=False)
            out = (x - mean[:, None]) / torch.sqrt(var[:, None] + self.eps)
            with torch.no_grad():
                n = x.size(1)
                var_u = var * n / max(n - 1, 1)
                for b in range(x.size(0)):
                    self.running_mean.mul_(1 - self.momentum).add_(mean[b], alpha=self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(var_u[b], alpha=self.momentum)
                self.num_batches_tracked += x.size(0)
        else:
            out = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return out * self.weight + self.bias


class BioBGT(nn.Module):
    def __init__(self, in_size, num_classes, d_model=128, num_heads=8, num_layers=3, dropout=0.1, batch_norm=True,
                 update_dim=128, ffn_mult=2, activation="relu"):
        super().__init__()
        self.embedding = nn.Linear(in_size, d_model, bias=False)
        self.embedding_ve = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([BioBGTLayer(d_model, num_heads, ffn_mult * d_model, dropout, batch_norm, activation, update_dim)  # upstream: FFN = 2 x hidden
                                     for _ in range(num_layers)])
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, num_classes))

    def forward(self, ts, ve, update_x):
        x = self.embedding(ts) + self.embedding_ve(ve.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x, update_x)
        return self.classifier(x).mean(dim=1)
