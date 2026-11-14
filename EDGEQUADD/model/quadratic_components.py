
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm, DropPath
from .basic_blocks import ASPP2d


class ASPPQuadraticAdapter(nn.Module):
    """
    Quadratic adapter with ASPP for multi-scale processing of the correlation matrix.

    Forward:
        h = RMSNorm(x)
        base = Linear(h)
        C' = ASPP(C)    # [B,1,N,N] -> [B,1,N,N]
        s = C' @ h      # [B,N,N] @ [B,N,d] -> [B,N,d]
        quad = Linear(h * s)    # rank 0; with rank r > 0: Wo(Ux(h) * Vc(s))
        gate = sigmoid(Linear(h) / temp)
        y = gate * (base + alpha * quad)
        out = x + DropPath( Dropout( act(y) ) )
    """
    def __init__(self, d, activation='gelu', dropout=0., droppath=0., rank=0, aspp_rates=(1, 2, 4),
                 use_quad=True, use_aspp=True, use_gate=True, quad_input='graph'):
        super().__init__()
        # ablation switches (all True = the model as published): quadratic term, ASPP filtering of corr, sigmoid gate
        self.use_quad, self.use_aspp, self.use_gate = bool(use_quad), bool(use_aspp), bool(use_gate)
        # second factor of the quadratic term: 'graph' = corr-mixed C'h (published), 'self' = h itself (QuadraNet V2 neuron),
        # 'graph_norm' = C'h / N (same mixing on the scale of one row), 'row' = the ROI's own correlation row (BQN-style)
        if quad_input not in ('graph', 'self', 'graph_norm', 'row'):
            raise ValueError(f"unknown quad_input {quad_input!r}")
        self.quad_input = quad_input
        self.norm = RMSNorm(d)
        self.base = nn.Linear(d, d, bias=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(dropout)
        self.dp = DropPath(droppath)

        self.gate = nn.Linear(d, d)
        nn.init.constant_(self.gate.bias, 0.0)
        self.gate_temp = nn.Parameter(torch.tensor(1.0))

        act_map = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}
        if str(activation).lower() not in act_map:
            raise ValueError(f"unknown activation {activation!r}; choose from {sorted(act_map)}")
        self.act = act_map[str(activation).lower()]()

        # ASPP for multi-scale filtering of corr
        self.aspp = ASPP2d(in_ch=1, out_ch=1, rates=aspp_rates, norm="gn", act=nn.GELU)

        # quadratic term: full projection (rank 0) or low-rank head (rank > 0)
        self.rank = int(rank) if rank is not None else 0
        if self.rank > 0:
            # low-rank quadratic head
            self.Ux = nn.Linear(d, self.rank, bias=False)
            self.Vc = nn.Linear(d, self.rank, bias=False)
            self.Wo = nn.Linear(self.rank, d, bias=False)
        else:
            self.quad = nn.Linear(d, d, bias=True)
            nn.init.normal_(self.quad.weight, 0, 1.0 / math.sqrt(d))
            nn.init.zeros_(self.quad.bias)

    def forward(self, x, corr):
        """
        x:    [B, N, d]
        corr: [B, N, N]  (square)
        """
        h = self.norm(x)
        base = self.base(h)

        if self.use_quad:
            if self.quad_input not in ('graph', 'self', 'graph_norm', 'row'):
                raise ValueError(f"unknown quad_input {self.quad_input!r}")
            if self.quad_input == 'self':
                s = h
            elif self.quad_input == 'row':
                s = corr                                             # [B, N, N]: row i = ROI i's correlation profile
            else:
                # ASPP over corr (ablation: the raw corr)
                corr_aspp = self.aspp(corr.unsqueeze(1)).squeeze(1) if self.use_aspp else corr  # [B, N, N]

                # Spatial mixing then quadratic
                s = torch.einsum('bni,bid->bnd', corr_aspp, h)       # [B, N, d]
                if self.quad_input == 'graph_norm':
                    s = s / corr.shape[-1]

            if self.rank > 0:
                ux = self.Ux(h)                                   # [B, N, r]
                vs = self.Vc(s)                                   # [B, N, r]
                quad = self.Wo(ux * vs)                           # [B, N, d]
            else:
                quad = self.quad(h * s)                           # [B, N, d]
            y = base + self.alpha * quad
        else:
            y = base

        if self.use_gate:
            y = torch.sigmoid(self.gate(h) / self.gate_temp.clamp_min(0.1)) * y

        y = self.drop(self.act(y))
        return x + self.dp(y)