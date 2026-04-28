"""DeepOHeat-style package-level operator surrogate."""

from __future__ import annotations

import torch
from torch import nn


class PackageThermalDeepONet(nn.Module):
    """G_theta(X, y) -> T(y) with CNN branch and coordinate trunk."""

    def __init__(
        self,
        in_channels: int,
        *,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        initial_temperature: float = 293.15,
    ) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.bias = nn.Parameter(torch.tensor(float(initial_temperature)))

    def forward(self, package_tensor: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        branch = self.branch(package_tensor)
        if coords.dim() == 2:
            trunk = self.trunk(coords)
            return torch.einsum("bf,nf->bn", branch, trunk) / branch.shape[-1] + self.bias
        if coords.dim() == 3:
            trunk = self.trunk(coords)
            return torch.einsum("bf,bnf->bn", branch, trunk) / branch.shape[-1] + self.bias
        raise ValueError("coords must have shape [N,2] or [B,N,2]")
