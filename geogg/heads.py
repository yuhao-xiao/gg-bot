"""Trainable classification heads that sit on top of frozen DINOv2 embeddings.

This is the only part we train in v1: a small MLP mapping a 768-d embedding to a
probability over the ~3,680 geocells.
"""

from __future__ import annotations

import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 1024, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskHead(nn.Module):
    """Shared trunk -> three heads: fine geocell + medium region + country.

    Hierarchical auxiliary supervision (ISN/PIGEON-style): the medium head
    (level-4 S2 ancestors of our cells) and the country head are training-time
    signals that push the shared trunk to learn coarse-to-fine geographic
    structure. This rewards 'close-but-not-exact' cell predictions implicitly:
    nearby cells share a medium parent and (usually) a country, so confusing
    them gets aux-loss credit. At inference we read the cell head."""

    def __init__(self, in_dim: int, n_cells: int, n_mediums: int, n_countries: int,
                 hidden: int = 1024, dropout: float = 0.3, trunk_layers: int = 1) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(trunk_layers - 1):  # extra hidden blocks for more capacity
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.cell = nn.Linear(hidden, n_cells)
        self.medium = nn.Linear(hidden, n_mediums)
        self.country = nn.Linear(hidden, n_countries)

    def forward(self, x):
        h = self.trunk(x)
        return self.cell(h), self.medium(h), self.country(h)
