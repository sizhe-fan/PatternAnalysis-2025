# PyTorch Siamese Network (两个共享分支 + 距离头)
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBackbone(nn.Module):
    """简单 CNN 作为共享特征提取器（可替换为 ResNet18）"""
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.normalize(x, dim=1)

class SiameseNet(nn.Module):
    """输出 pair 的相似度概率（0=不同类, 1=同类/恶性一致性可按任务定义）"""
    def __init__(self, feat_dim=128, distance="l2"):
        super().__init__()
        self.backbone = ConvBackbone(feat_dim=feat_dim)
        self.distance = distance
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _pair_features(self, f1, f2):
        if self.distance == "l1":
            d = torch.abs(f1 - f2)
        elif self.distance == "cos":
            d = 1 - F.cosine_similarity(f1, f2).unsqueeze(1)
        else:  # "l2"
            d = torch.sqrt(torch.sum((f1 - f2) ** 2, dim=1, keepdim=True) + 1e-8)
        return d

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        d = self._pair_features(f1, f2)
        logits = self.head(d)          # shape: (B,1)
        return logits.squeeze(1)       # -> (B,)
