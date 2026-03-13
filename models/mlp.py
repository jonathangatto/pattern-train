"""
MLP classifier on top of MERT frame embeddings.

Applies mean+max temporal pooling over the T frame dimension to collapse
(B, T, D) → (B, 2D) and then classifies with a 3-layer MLP.

This is the simplest baseline: all temporal ordering information is discarded
after pooling.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Input:  (B, T, D)  — batch of MERT frame sequences
    Output: (B, n_labels)  — raw logits (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        in_dim  : int,
        n_labels: int,
        dropout : float = 0.3,
        **kwargs,          # absorb unrecognised kwargs from build_model
    ):
        super().__init__()
        pool_dim = in_dim * 2   # mean + max concatenation

        self.net = nn.Sequential(
            nn.Linear(pool_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        mean_pool = x.mean(dim=1)           # (B, D)
        max_pool  = x.max(dim=1).values     # (B, D)
        pooled    = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2D)
        return self.net(pooled)
