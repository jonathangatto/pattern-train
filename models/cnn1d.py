"""
1D residual CNN classifier over MERT frame embeddings.

Rather than immediately pooling away the temporal dimension, this model
convolves across the ~75 time frames produced by MERT for a 1-second clip,
learning local temporal patterns before aggregating.

Architecture:
    (B, T, D) → permute → (B, D, T)
        → Conv1d projection (D → channels)
        → N × ResBlock1D (channels → channels)
        → global mean+max pool → (B, 2*channels)
        → MLP head → (B, n_labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Two-layer 1D residual block with BatchNorm and ReLU."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class CNN1DClassifier(nn.Module):
    """
    Input:  (B, T, D)  — batch of MERT frame sequences (T ≈ 75 for 1-sec clips)
    Output: (B, n_labels)  — raw logits (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        in_dim   : int,
        n_labels : int,
        channels : int   = 256,
        n_blocks : int   = 3,
        dropout  : float = 0.3,
        **kwargs,          # absorb unrecognised kwargs from build_model
    ):
        super().__init__()

        # Normalise MERT frame embeddings — they have large per-dim variance
        # that causes the conv projection to blow up before BN stabilises.
        self.input_norm = nn.LayerNorm(in_dim)

        # 1×1 conv to project from MERT hidden dim to working channel width
        self.proj    = nn.Conv1d(in_dim, channels, kernel_size=1)
        self.bn_proj = nn.BatchNorm1d(channels)

        self.blocks = nn.Sequential(*[ResBlock1D(channels) for _ in range(n_blocks)])

        # Classification head after global pooling
        self.head = nn.Sequential(
            nn.Linear(channels * 2, 128),  # *2 for mean+max
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → normalise → (B, D, T) for Conv1d
        x = self.input_norm(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn_proj(self.proj(x)))  # (B, C, T)
        x = self.blocks(x)                       # (B, C, T)

        mean_pool = x.mean(dim=2)                # (B, C)
        max_pool  = x.max(dim=2).values          # (B, C)
        x = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2C)
        return self.head(x)
