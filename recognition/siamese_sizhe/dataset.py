# dataset.py
"""
Builds paired samples (img1, img2, label) from:
- CSV: data/train.csv
- Images: data/train/*.jpg

Only binary labels 0/1 are accepted (pulled from columns: target/label/malignant).
Rows with non-binary or missing labels are skipped.
"""

import os
import csv
import random
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PairDataset(Dataset):
    """
    Dataset that returns random positive/negative pairs for Siamese training.

    CSV requirements:
      - One of the following image id columns must exist:
        ["image_name", "image_id", "image", "isic_id"]
      - One of the following binary label columns must exist:
        ["target", "label", "malignant"] with values in {0, 1}

    Each __getitem__ randomly samples a 50/50 mix of positive and negative pairs.

    Args:
        root (str): Directory containing the CSV and image folder.
        csv_name (str): CSV filename (default: "train.csv").
        img_dir (str): Subfolder containing images (default: "train").
        image_size (int): Square resize for images.
        mode (str): Unused placeholder for compatibility ("train"/"val"/"test").
        seed (int): RNG seed for reproducible pairing.
    """
    def __init__(
        self,
        root: str,
        csv_name: str = "train.csv",
        img_dir: str = "train",
        image_size: int = 224,
        mode: str = "train",
        seed: int = 42,
    ):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.csv_path = os.path.join(root, csv_name)
        self.mode = mode
        self.rng = random.Random(seed)

        # Parse CSV: (image_path, label) where label ∈ {0, 1}
        self.items: List[Tuple[str, int]] = []
        with open(self.csv_path, newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                image_id = (
                    row.get("image_name")
                    or row.get("image_id")
                    or row.get("image")
                    or row.get("isic_id")
                )
                if not image_id:
                    continue

                raw = row.get("target") or row.get("label") or row.get("malignant")
                try:
                    y = int(raw)
                except Exception:
                    # Skip rows with non-integer or missing labels
                    continue
                if y not in (0, 1):
                    # Keep only binary samples
                    continue

                p = os.path.join(self.img_dir, f"{image_id}.jpg")
                if os.path.exists(p):
                    self.items.append((p, y))

        # Split into positive/negative pools for fast sampling
        self.pos_pool = [it for it in self.items if it[1] == 1]
        self.neg_pool = [it for it in self.items if it[1] == 0]
        assert len(self.pos_pool) > 0 and len(self.neg_pool) > 0, (
            "Dataset must contain both positive and negative samples. "
            "Check your CSV/image paths and label columns."
        )

        # Basic preprocessing (matches the config that achieved ~0.82–0.83 AUC)
        self.tf = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # Length controls how many pairs are drawn per epoch
        self.length = len(self.items)

    def __len__(self) -> int:
        return self.length

    def _sample_pos_pair(self) -> Tuple[str, str, int]:
        """Sample a positive pair: two images with label=1 (same class)."""
        a = self.rng.choice(self.pos_pool)
        b = self.rng.choice(self.pos_pool)
        return a[0], b[0], 1

    def _sample_neg_pair(self) -> Tuple[str, str, int]:
        """Sample a negative pair: one positive and one negative image (different class)."""
        a = self.rng.choice(self.pos_pool)
        b = self.rng.choice(self.neg_pool)
        return a[0], b[0], 0

    def __getitem__(self, idx: int):
        """
        Return a randomly sampled pair.

        Returns:
            x1 (Tensor): First image tensor (3, H, W)
            x2 (Tensor): Second image tensor (3, H, W)
            y  (Tensor): Float label tensor scalar (1.0 for positive pair, 0.0 for negative)
        """
        # 50% positive pairs, 50% negative pairs
        if self.rng.random() < 0.5:
            p1, p2, y = self._sample_pos_pair()
        else:
            p1, p2, y = self._sample_neg_pair()

        x1 = self.tf(Image.open(p1).convert("RGB"))
        x2 = self.tf(Image.open(p2).convert("RGB"))
        return x1, x2, torch.tensor(y, dtype=torch.float32)
