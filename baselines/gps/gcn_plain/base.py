from abc import abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor) -> torch.tensor:
        pass


def build_edge_index(node_feature, threshold):
    """Batched edge index from thresholded correlation matrices; the same graph as `dense_to_batch_graph` in
    baselines/gcn/model.py (tunedGNN family).

    For each subject b a directed edge i -> j wherever node_feature[b, i, j] > threshold, diagonal excluded, each directed
    edge once (the matrix is symmetric, so both directions are present), node ids offset by b * num_nodes. Edges are
    unweighted. A subject without edges simply has none: GCNConv adds its own self-loops and attention does not need edges.
    """
    batch_size, num_nodes, _ = node_feature.shape
    mask = node_feature > threshold
    mask = mask & ~torch.eye(num_nodes, dtype=torch.bool, device=node_feature.device).unsqueeze(0)
    b, i, j = torch.nonzero(mask, as_tuple=True)
    offset = b * num_nodes
    return torch.stack([i + offset, j + offset])
