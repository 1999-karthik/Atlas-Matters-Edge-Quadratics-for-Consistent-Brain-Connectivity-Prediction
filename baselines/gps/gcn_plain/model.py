import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from omegaconf import DictConfig
from .base import BaseModel, build_edge_index

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}


def make_activation(name):
    """Activation module by name; raises ValueError for an unknown name."""
    try:
        return ACTIVATIONS[str(name).lower()]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}") from None


class GCN(BaseModel):
    """
    Graph Convolutional Network for brain network classification.
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.node_num = cfg.dataset.node_sz
        self.node_feature_sz = cfg.dataset.node_feature_sz
        self.hidden_dim = getattr(cfg.model, 'hidden_dim', 64)
        self.num_layers = getattr(cfg.model, 'num_layers', 3)
        self.dropout = getattr(cfg.model, 'dropout', 0.5)
        self.act = make_activation(getattr(cfg.model, 'activation', 'relu'))  # activation of the GCN layers
        self.pooling = getattr(cfg.model, 'pooling', 'mean')
        self.threshold = getattr(cfg.model, 'threshold', 0.3)  # edge threshold of the graph
        
        # Number of classes: explicit cfg.dataset.num_classes wins, otherwise PPMI -> 4, else binary
        explicit = getattr(cfg.dataset, 'num_classes', None)
        if explicit is not None:
            self.num_classes = int(explicit)
        else:
            self.num_classes = 4 if cfg.dataset.name.lower().startswith('ppmi') else 2
        
        # Input projection layer
        self.input_proj = nn.Linear(self.node_feature_sz, self.hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            else:
                self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, self.num_classes)
        )
        
    def forward(self, time_series, node_feature):
        batch_size, num_nodes, _ = node_feature.shape
        
        # Use correlation matrix as node features (each row is a node's correlations)
        x = node_feature.view(-1, self.node_feature_sz)
        x = self.input_proj(x)

        edge_index = build_edge_index(node_feature, self.threshold)
        
        # GCN layers
        for i, (gcn_layer, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn_layer(x, edge_index)
            x = bn(x)
            x = self.act(x)
            x = self.dropout_layer(x)
        
        # Reshape back to [batch_size, num_nodes, hidden_dim]
        x = x.view(batch_size, num_nodes, self.hidden_dim)
        
        # Global pooling - use simple mean/max/sum across nodes
        if self.pooling == 'mean':
            x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        elif self.pooling == 'max':
            x, _ = torch.max(x, dim=1)  # [batch_size, hidden_dim]
        elif self.pooling == 'add':
            x = torch.sum(x, dim=1)  # [batch_size, hidden_dim]
        else:
            # Default to mean pooling
            x = torch.mean(x, dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_attention_weights(self):
        """GCN doesn't have attention weights, return None"""
        return None
