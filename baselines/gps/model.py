import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch.nn import ModuleList, Linear, BatchNorm1d, Sequential, ReLU
from omegaconf import DictConfig
from typing import Dict, Any, Optional
from .base import BaseModel, build_edge_index
from .rrwp import add_full_rrwp


class GPS(BaseModel):
    """
    Graph Positional Encoding + GPSConv for brain network classification.
    This model uses PyTorch Geometric's GPSConv with RRWP positional encoding.
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.node_num = cfg.dataset.node_sz
        self.node_feature_sz = cfg.dataset.node_feature_sz
        self.channels = getattr(cfg.model, 'channels', 64)
        self.pe_dim = getattr(cfg.model, 'pe_dim', 32)
        self.num_layers = getattr(cfg.model, 'num_layers', 3)
        self.dropout = getattr(cfg.model, 'dropout', 0.5)
        self.pooling = getattr(cfg.model, 'pooling', 'add')  # add, mean, max
        self.use_positional_encoding = getattr(cfg.model, 'use_positional_encoding', True)
        self.threshold = getattr(cfg.model, 'threshold', 0.3)  # edge threshold of the graph
        self.walk_length = getattr(cfg.model, 'walk_length', 8)  # RRWP walk length
        self.attn_type = getattr(cfg.model, 'attn_type', 'multihead')
        self.heads = getattr(cfg.model, 'heads', 4)
        self.activation = getattr(cfg.model, 'activation', 'relu')  # activation of the GPSConv feed-forward blocks
        if self.activation not in ('relu', 'gelu', 'elu', 'leaky_relu'):
            raise ValueError(f"unknown activation {self.activation!r}; choose from relu, gelu, elu, leaky_relu")
        
        # Number of classes: explicit cfg.dataset.num_classes wins, otherwise PPMI -> 4, else binary
        explicit = getattr(cfg.dataset, 'num_classes', None)
        if explicit is not None:
            self.num_classes = int(explicit)
        else:
            self.num_classes = 4 if cfg.dataset.name.lower().startswith('ppmi') else 2
        
        # Input projection for correlation matrix features
        self.input_proj = nn.Linear(self.node_feature_sz, self.channels - self.pe_dim)
        
        # Positional encoding components
        if self.use_positional_encoding:
            self.pe_lin = Linear(self.walk_length, self.pe_dim)
            self.pe_norm = BatchNorm1d(self.walk_length)
        
        # GPSConv layers
        self.convs = ModuleList()
        for _ in range(self.num_layers):
            # Create GPSConv layer with GCN as local message passing
            conv = GPSConv(
                channels=self.channels,
                conv=GCNConv(self.channels, self.channels),
                heads=self.heads,
                act=self.activation,
                dropout=self.dropout,
                attn_type=self.attn_type,
                attn_kwargs={}
            )
            self.convs.append(conv)
        
        # Classification head
        self.mlp = Sequential(
            Linear(self.channels, self.channels // 2),
            ReLU(),
            Linear(self.channels // 2, self.channels // 4),
            ReLU(),
            Linear(self.channels // 4, self.num_classes)
        )
        
        
    def forward(self, time_series, node_feature):
        """
        Forward pass for GPS using GPSConv.
        
        Args:
            time_series: Time series data (not used in GPS)
            node_feature: Correlation matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size, num_nodes, _ = node_feature.shape
        device = node_feature.device
        
        # Use correlation matrix as node features (each row is a node's correlations)
        x = node_feature.view(-1, self.node_feature_sz)
        x = self.input_proj(x)
        
        # Add RRWP positional encoding if enabled
        if self.use_positional_encoding:
            # Use the correlation matrix (node_feature) as adjacency matrix for PE computation
            # same edge set as the message-passing graph: the diagonal (self-correlation 1) is not an edge
            no_diag = node_feature * (1 - torch.eye(num_nodes, device=device, dtype=node_feature.dtype)).unsqueeze(0)
            pe = add_full_rrwp(no_diag, self.walk_length, threshold=self.threshold)
            # Normalize positional encoding
            pe_flat = pe.view(-1, self.walk_length)
            x_pe = self.pe_norm(pe_flat)
            x_pe = self.pe_lin(x_pe)
            
            # Combine node features and positional encoding
            x = torch.cat([x, x_pe], dim=1)
        else:
            # Pad with zeros to match expected dimension
            x = torch.cat([x, torch.zeros(x.size(0), self.pe_dim, device=device)], dim=1)
        
        # Create edge index from correlation matrix
        edge_index = build_edge_index(node_feature, self.threshold)
        batch = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        
        # GPSConv layers - using GCNConv as local message passing
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        
        # Global pooling using batch information
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:  # default to add
            x = global_add_pool(x, batch)
        
        # Classification
        logits = self.mlp(x)
        
        return logits
    
    def get_attention_weights(self):
        """GPSConv doesn't provide attention weights, return None"""
        return None
