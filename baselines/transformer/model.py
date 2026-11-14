import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from omegaconf import DictConfig
import torch.nn.functional as F
from .base import BaseModel, pick_num_heads

ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu, "elu": F.elu, "leaky_relu": F.leaky_relu}


class GraphTransformer(BaseModel):

    def __init__(self, cfg: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        self.readout = cfg.model.readout
        self.node_num = cfg.dataset.node_sz

        for _ in range(cfg.model.self_attention_layer):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=cfg.dataset.node_feature_sz,
                                        nhead=pick_num_heads(cfg.dataset.node_feature_sz, cfg.model.get('heads', 4)),  # upstream: 4
                                        dim_feedforward=cfg.model.get('ffn_dim', 1024),  # upstream: 1024
                                        dropout=cfg.model.get('dropout', 0.1),  # upstream: nn.TransformerEncoderLayer default 0.1
                                        activation=ACTIVATIONS[cfg.model.get('activation', 'relu')],  # upstream: relu
                                        batch_first=True)
            )

        final_dim = cfg.dataset.node_feature_sz

        if self.readout == "concat":
            self.dim_reduction = nn.Sequential(
                nn.Linear(cfg.dataset.node_feature_sz, 8),
                nn.LeakyReLU()
            )
            final_dim = 8 * self.node_num

        elif self.readout == "sum":
            self.norm = nn.BatchNorm1d(cfg.dataset.node_feature_sz)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, getattr(cfg.dataset, 'num_classes', 2))
        )

    def forward(self, time_seires, node_feature):
        bz, _, _, = node_feature.shape

        for atten in self.attention_list:
            node_feature = atten(node_feature)

        if self.readout == "concat":
            node_feature = self.dim_reduction(node_feature)
            node_feature = node_feature.reshape((bz, -1))

        elif self.readout == "mean":
            node_feature = torch.mean(node_feature, dim=1)
        elif self.readout == "max":
            node_feature, _ = torch.max(node_feature, dim=1)
        elif self.readout == "sum":
            node_feature = torch.sum(node_feature, dim=1)
            node_feature = self.norm(node_feature)

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]
