import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from .base import BaseModel


# activation of the hidden layers (E2E blocks, E2N, N2G, dense1, dense2); 0.33 is the leaky slope upstream uses everywhere
ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu, "elu": F.elu, "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.33)}


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.in_planes = 1
        self.d = config.dataset.node_sz

        # Upstream widths are 32 / 64 / 256 / 128 / 30 and dropout 0.5 (hard-coded there). `width_mult` scales all widths
        # and `dropout` sets p; the defaults 1.0 / 0.5 reproduce the upstream network exactly.
        w = float(config.model.get('width_mult', 1.0))
        self.dropout = float(config.model.get('dropout', 0.5))
        self.act = ACTIVATIONS[config.model.get('activation', 'leaky_relu')]
        c1, c2, g, d1, d2 = (max(1, round(c * w)) for c in (32, 64, 256, 128, 30))
        self.e2econv1 = E2EBlock(1, c1, config.dataset.node_sz, bias=True)
        self.e2econv2 = E2EBlock(c1, c2, config.dataset.node_sz, bias=True)
        self.E2N = torch.nn.Conv2d(c2, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, g, (self.d, 1))
        self.dense1 = torch.nn.Linear(g, d1)
        self.dense2 = torch.nn.Linear(d1, d2)
        self.dense3 = torch.nn.Linear(d2, getattr(config.dataset, 'num_classes', 2))

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):
        node_feature = node_feature.unsqueeze(dim=1)
        out = self.act(self.e2econv1(node_feature))
        out = self.act(self.e2econv2(out))
        out = self.act(self.E2N(out))
        out = F.dropout(self.act(self.N2G(out)), p=self.dropout, training=self.training)
        out = out.view(out.size(0), -1)
        out = F.dropout(self.act(self.dense1(out)), p=self.dropout, training=self.training)
        out = F.dropout(self.act(self.dense2(out)), p=self.dropout, training=self.training)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)  # upstream applies this to the logits too; kept fixed

        return out
