"""
modules.py

Defines the architecture of the Siamese Network model.
Includes:
- ConvBackbone: a simple CNN feature extractor (can be replaced with ResNet)
- SiameseNet: the full Siamese network for similarity prediction
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBackbone(nn.Module):
    """
    A lightweight CNN backbone used for feature extraction.
    Each image passes through identical convolutional layers
    to produce a compact embedding vector.

    Args:
        in_ch (int): Number of input channels (default: 3 for RGB images).
        feat_dim (int): Output feature dimension of the embedding.
    """
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN backbone.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Normalized feature vector of shape (B, feat_dim)
        """
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.normalize(x, dim=1)


class SiameseNet(nn.Module):
    """
    Siamese Network that predicts similarity between two images.
    Takes two input images, extracts their embeddings via shared backbone,
    computes distance features, and outputs a similarity probability.

    Args:
        feat_dim (int): Feature dimension from the backbone.
        distance (str): Distance metric to use ("l2" or "abs").
    """
    def __init__(self, feat_dim=128, distance="l2"):
        super().__init__()
        self.backbone = ConvBackbone(feat_dim=feat_dim)
        self.distance = distance
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _pair_features(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise feature difference between two embeddings.

        Args:
            f1, f2 (torch.Tensor): Feature vectors of shape (B, feat_dim)

        Returns:
            torch.Tensor: Element-wise absolute difference (B, feat_dim)
        """
        return torch.abs(f1 - f2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a pair of images.

        Args:
            x1, x2 (torch.Tensor): Input images of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Similarity logits (B,)
        """
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        d = self._pair_features(f1, f2)
        logits = self.head(d)  # (B, 1)
        return logits.squeeze(1)  # -> (B,)

