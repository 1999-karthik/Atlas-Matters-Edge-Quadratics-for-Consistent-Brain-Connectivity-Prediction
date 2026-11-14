import torch
import torch.nn as nn

def _make_norm(norm, num_channels, groups=8):
    """Create normalization layer based on type."""
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        # Find the largest divisor of num_channels that is <= groups
        g = min(groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    else:
        raise ValueError("norm must be 'bn' or 'gn'")



class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(x.dtype)
        return x * mask / keep