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


def pick_num_heads(d_model: int, preferred: int = 4) -> int:
    """Largest head count <= preferred that divides d_model (nn.MultiheadAttention needs d_model % heads == 0)."""
    for h in range(preferred, 0, -1):
        if d_model % h == 0:
            return h
    return 1
