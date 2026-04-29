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
        feature_dim: int | None = 64,
        branch_dim: int | None = None,
        trunk_dim: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        initial_temperature: float = 293.15,
    ) -> None:
        super().__init__()
        if branch_dim is None:
            branch_dim = int(feature_dim or 64)
        if trunk_dim is None:
            trunk_dim = int(feature_dim or branch_dim)
        if branch_dim != trunk_dim:
            raise ValueError("branch_dim and trunk_dim must match for DeepONet product")

        branch_layers: list[nn.Module] = [
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
        ]
        if use_batchnorm:
            branch_layers.append(nn.BatchNorm2d(24))
        branch_layers.extend(
            [
                nn.SiLU(),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
            ]
        )
        if use_batchnorm:
            branch_layers.append(nn.BatchNorm2d(48))
        branch_layers.extend(
            [
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(48 * 4 * 4, hidden_dim),
                nn.SiLU(),
            ]
        )
        if dropout > 0.0:
            branch_layers.append(nn.Dropout(dropout))
        branch_layers.append(nn.Linear(hidden_dim, branch_dim))
        self.branch = nn.Sequential(
            *branch_layers,
        )

        trunk_layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(max(1, num_layers)):
            trunk_layers.append(nn.Linear(in_dim, hidden_dim))
            trunk_layers.append(nn.SiLU())
            if dropout > 0.0:
                trunk_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        trunk_layers.append(nn.Linear(hidden_dim, trunk_dim))
        self.trunk = nn.Sequential(*trunk_layers)
        self.output_dim = int(branch_dim)
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
