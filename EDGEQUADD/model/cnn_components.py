import torch
import torch.nn as nn
from .normalization import _make_norm
from .basic_blocks import ASPP2d, ResBlock


class SimpleCNNCorrExtractor(nn.Module):
    """Enhanced CNN extractor using ResBlocks and ASPP for better feature learning."""
    def __init__(self, in_ch=1, base_ch=16, out_dim=64, diag_mode="zero",
                 symmetrize=True, norm="bn", dropout=0.0, aspp_rates=(1,2,4)):
        super().__init__()
        stem_in = in_ch if diag_mode != "channel" else in_ch + 1
        self.diag_mode = diag_mode
        self.symmetrize = symmetrize
        
        # Initial 1x1 convolution to adjust channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(stem_in, base_ch, kernel_size=1, bias=False),
            _make_norm(norm, base_ch),
            nn.GELU(),
        )
        
        # ResBlocks for better feature learning
        self.res_blocks = nn.Sequential(
            ResBlock(base_ch, base_ch, stride=1, norm=norm),
            ResBlock(base_ch, base_ch*2, stride=1, norm=norm),
        )
        
        # ASPP for multi-scale processing
        self.aspp = ASPP2d(base_ch*2, base_ch*2, rates=aspp_rates, norm=norm, act=nn.GELU)
        
        # Final ResBlock for feature refinement
        self.final_res = ResBlock(base_ch*2, base_ch*4, stride=1, norm=norm)
        
        self.drop = nn.Dropout2d(dropout)
        self.channels = base_ch*4
        self.proj = nn.Linear(self.channels, out_dim, bias=True)
        
        # Add skip connection for residual
        self.skip_to_proj = nn.Sequential(
            nn.Conv2d(stem_in, base_ch*4, 1, bias=False),
            _make_norm(norm, base_ch*4),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Xavier initialization for GELU activation (better than Kaiming)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _prep_input(self, x):
        if self.symmetrize:
            x = 0.5 * (x + x.transpose(-1, -2))
        if self.diag_mode == "keep":
            return x
        B, C, N, _ = x.shape
        diag = x.diagonal(dim1=-2, dim2=-1)           
        if self.diag_mode == "zero":
            x = x - torch.diag_embed(diag)           
            return x
        else:  # "channel"
            x_strip = x - torch.diag_embed(diag)
            # average diag across channels -> (B, 1, N)
            diag_mean = diag.mean(dim=1, keepdim=True)        # (B, 1, N)
            # broadcast to NxN map
            diag_chan = diag_mean.unsqueeze(-1).expand(B, 1, N, N)  # (B, 1, N, N)
            return torch.cat([x_strip, diag_chan], dim=1)

    def forward(self, corr_mat_4d, target_dim_ignored=None):
        x = self._prep_input(corr_mat_4d)
        identity = self.skip_to_proj(x)  # match channels for residual
        
        # Enhanced feature extraction with ResBlocks
        fmap = self.initial_conv(x)     # Initial 1x1 conv
        fmap = self.res_blocks(fmap)    # ResBlocks for better feature learning
        fmap = self.aspp(fmap)          # ASPP multi-scale processing
        fmap = self.final_res(fmap)     # Final ResBlock for refinement
        fmap = self.drop(fmap)          # Dropout
        fmap = fmap + identity          # Residual connection
        
        # Global average pooling to get node features
        row_pool = fmap.mean(dim=3)
        col_pool = fmap.mean(dim=2)
        node_feat = 0.5 * (row_pool + col_pool)           
        node_feat = node_feat.transpose(1, 2)             
        return self.proj(node_feat)                      