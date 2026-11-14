"""
Relative random-walk positional encoding (RRWP; Ma et al., "Graph Inductive Biases in Transformers without Message
Passing", ICML 2023).
"""

import torch
import torch.nn.functional as F
from typing import Optional


def add_full_rrwp(data: torch.Tensor, walk_length: int, threshold: float = 0.0) -> torch.Tensor:
    """
    Compute RRWP for a batch of adjacency matrices.
    
    Args:
        data: Batch of adjacency matrices of shape (batch_size, num_nodes, num_nodes)
        walk_length: Length of random walks
        
    Returns:
        Batch of positional encodings of shape (batch_size, num_nodes, walk_length)
    """
    batch_size, num_nodes, _ = data.shape
    device = data.device
    
    pes = []
    for batch_idx in range(batch_size):
        adj_matrix = data[batch_idx]
        pe = add_every_rrwp(adj_matrix, walk_length, threshold=threshold)
        pes.append(pe)
    
    return torch.stack(pes, dim=0)


def add_every_rrwp(data: torch.Tensor,
                   walk_length: int = 8,
                   threshold: float = 0.3,
                   add_identity: bool = True,
                   spd: bool = False,
                   **kwargs) -> torch.Tensor:
    """
    Compute Random Walk with Restart Positional Encoding (RRWP) for graph nodes.
    
    Args:
        data: Adjacency matrix of shape (num_nodes, num_nodes)
        walk_length: Length of random walks
        threshold: Threshold for edge filtering
        add_identity: Whether to include identity matrix
        spd: Whether to use shortest path distance encoding
        
    Returns:
        Positional encoding matrix of shape (num_nodes, walk_length)
    """
    device = data.device
    num_nodes = data.shape[0]
    
    # Create edge index from adjacency matrix above threshold
    edge_mask = data > threshold
    edge_index = torch.column_stack(torch.where(edge_mask)).T.contiguous()
    
    if edge_index.size(0) == 0:
        # Handle case with no edges above threshold
        return torch.zeros(num_nodes, walk_length, device=device)
    
    # Extract edge weights
    edge_weights = data[edge_index[0], edge_index[1]]
    
    # Create sparse adjacency matrix
    adj_sparse = torch.sparse_coo_tensor(
        edge_index, edge_weights, (num_nodes, num_nodes)
    )
    adj = adj_sparse.to_dense()
    
    # Compute normalized adjacency matrix D^{-1} A
    degree = adj.sum(dim=1)
    degree_inv = torch.where(degree > 0, 1.0 / degree, 0.0)
    adj_normalized = adj * degree_inv.view(-1, 1)
    
    # Initialize positional encoding list
    pe_list = []
    
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=device))
    
    pe_list.append(adj_normalized)
    
    # Compute higher-order powers of adjacency matrix
    current_power = adj_normalized
    for _ in range(2, walk_length):
        current_power = current_power @ adj_normalized
        pe_list.append(current_power)
    
    # Stack and extract diagonal (node-level features)
    pe_tensor = torch.stack(pe_list, dim=-1)  # (num_nodes, num_nodes, walk_length)
    abs_pe = pe_tensor.diagonal().transpose(0, 1)  # (num_nodes, walk_length)
    
    return abs_pe
