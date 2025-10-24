# dataset.py
# 读取 data/train.csv 与 data/train/*.jpg，生成成对样本 (img1, img2, label)
# 仅接受二分类标签 0/1（来自 target/label/malignant）；非 0/1 的行会被跳过
import os, csv, random
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PairDataset(Dataset):
    def __init__(self, root: str, csv_name: str = "train.csv", img_dir: str = "train",
                 image_size: int = 224, mode: str = "train", seed: int = 42):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.csv_path = os.path.join(root, csv_name)
        self.mode = mode
        self.rng = random.Random(seed)

        # 读取 csv：image_name, target/label/malignant（仅 0/1）
        self.items: List[Tuple[str, int]] = []  # (path, y)
        with open(self.csv_path, newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                image_id = row.get("image_name") or row.get("image_id") or row.get("image") or row.get("isic_id")
                if not image_id:
                    continue
                raw = row.get("target") or row.get("label") or row.get("malignant")
                try:
                    y = int(raw)
                except Exception:
                    # 对非常规取值直接跳过
                    continue
                if y not in (0, 1):
                    # 只保留二分类样本
                    continue
                p = os.path.join(self.img_dir, f"{image_id}.jpg")
                if os.path.exists(p):
                    self.items.append((p, y))

        # 分桶（正负样本池）
        self.pos_pool = [p for p in self.items if p[1] == 1]
        self.neg_pool = [p for p in self.items if p[1] == 0]
        assert len(self.pos_pool) > 0 and len(self.neg_pool) > 0, "需要包含正负样本（检查 CSV 与图片是否匹配）"

        # 预处理（与之前能到 0.82~0.83 的设置一致）
        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.length = len(self.items)  # 每个 epoch 采样与图片数同量级的配对

    def __len__(self):
        return self.length

    def _sample_pos_pair(self):
        a = self.rng.choice(self.pos_pool)
        b = self.rng.choice(self.pos_pool)
        return a[0], b[0], 1  # 同类=1

    def _sample_neg_pair(self):
        a = self.rng.choice(self.pos_pool)
        b = self.rng.choice(self.neg_pool)
        return a[0], b[0], 0  # 异类=0

    def __getitem__(self, idx):
        # 随机一半正对，一半负对
        if self.rng.random() < 0.5:
            p1, p2, y = self._sample_pos_pair()
        else:
            p1, p2, y = self._sample_neg_pair()

        x1 = self.tf(Image.open(p1).convert("RGB"))
        x2 = self.tf(Image.open(p2).convert("RGB"))
        return x1, x2, torch.tensor(y, dtype=torch.float32)
