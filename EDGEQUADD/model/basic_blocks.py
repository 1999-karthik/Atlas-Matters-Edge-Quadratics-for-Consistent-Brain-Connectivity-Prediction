"""
Convolutional building blocks (residual block, ASPP).
"""
import torch
import torch.nn as nn
from .normalization import _make_norm


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm="bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = _make_norm(norm, out_ch)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = _make_norm(norm, out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _make_norm(norm, out_ch)
            )
        
        # Initialize weights properly for GELU activation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly for GELU activation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Xavier/LeCun initialization for GELU (better than Kaiming for GELU)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        idt = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            idt = self.down(idt)
        out = self.act(out + idt)
        return out

class ASPP2d(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP): parallel dilated convolutions for multi-scale filtering.

    This block is linear; the quadratic interaction is in ASPPQuadraticAdapter.
    """
    def __init__(self, in_ch, out_ch, rates=(1, 2, 4), norm="bn", act=nn.GELU):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                _make_norm(norm, out_ch),
                act(),
            ) for r in rates
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, 1, bias=False),
            _make_norm(norm, out_ch),
            act(),
        )
        # Add a learnable skip when channels differ
        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _make_norm(norm, out_ch),
        )
        
        # Initialize weights properly for GELU activation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly for GELU activation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Xavier/LeCun initialization for GELU
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = self.skip(x)
        xs = [b(x) for b in self.branches]
        x = torch.cat(xs, dim=1)
        x = self.proj(x)
        return x + identity