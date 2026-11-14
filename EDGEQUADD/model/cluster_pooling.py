import torch
import torch.nn as nn
from typing import Tuple
from typing import Optional



class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
        orthogonal=True,
        freeze_center=True,
        project_assignment=True
    ) -> None:
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension  # forward dim
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.project_assignment = project_assignment
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)

        else:
            # Ensure proper type/device robustness
            initial_cluster_centers = cluster_centers.to(dtype=torch.float32)

        if orthogonal:
            eps = 1e-12
            device, dtype = initial_cluster_centers.device, initial_cluster_centers.dtype
            orth = torch.zeros_like(initial_cluster_centers, device=device, dtype=dtype)
            
            # start with a normalized first vector
            v0 = initial_cluster_centers[0]
            n0 = torch.norm(v0, p=2).clamp_min(eps)
            orth[0] = v0 / n0
            
            for i in range(1, cluster_number):
                vi = initial_cluster_centers[i]
                # Gram-Schmidt with epsilon guards
                for j in range(i):
                    u = orth[j]
                    num = torch.dot(u, vi)
                    den = torch.dot(u, u).clamp_min(eps)
                    vi = vi - (num / den) * u
                ni = torch.norm(vi, p=2).clamp_min(eps)
                orth[i] = vi / ni

            initial_cluster_centers = orth

        self.cluster_centers = nn.Parameter(
            initial_cluster_centers, requires_grad=(not freeze_center))

    @staticmethod
    def project(u, v, eps: float = 1e-12):
        den = torch.dot(u, u).clamp_min(eps)
        return (torch.dot(u, v) / den) * u

    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        if self.project_assignment:
            sim = batch @ self.cluster_centers.T                 # [*, K]
            # Energy-like scaling: square the similarity and divide by ||center||²
            denom = torch.norm(self.cluster_centers, dim=-1).pow(2).clamp_min(1e-12)
            sim = sim.pow(2) / denom
            return nn.functional.softmax(sim, dim=-1)

        else:

            norm_squared = torch.sum(
                (batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power
            return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def get_cluster_centers(self) -> torch.Tensor:
        return self.cluster_centers

class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        orthogonal=True,
        freeze_center=True, project_assignment=True
    ):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension  # forward dim
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha, orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment
        )

        self.loss_fn = nn.KLDivLoss(reduction='sum')

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, M = batch.shape
        assert N == M, f"DEC expects square (N×N) per sample, got {N}×{M}"
        
        node_num = N
        batch_size = B
        flattened_batch = batch.reshape(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        
        # Enforce encoder output shape contract
        expected = (batch_size, node_num * self.hidden_dimension)
        assert encoded.shape == expected, \
            f"Encoder must return shape {expected}, got {tuple(encoded.shape)}"
        
        encoded = encoded.view(batch_size * node_num, -1)
        assignment = self.assignment(encoded)
        assignment = assignment.view(batch_size, node_num, -1)
        encoded = encoded.view(batch_size, node_num, -1)
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment

    def target_distribution(self, batch: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        weight = (batch ** 2) / torch.sum(batch, 0, keepdim=False).clamp_min(eps)
        return (weight.t() / weight.sum(1, keepdim=True).clamp_min(eps)).t()

    def loss(self, assignment):
        flattened_assignment = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(flattened_assignment.log(), target) / flattened_assignment.size(0)

    def get_cluster_centers(self) -> torch.Tensor:
        return self.assignment.get_cluster_centers()


