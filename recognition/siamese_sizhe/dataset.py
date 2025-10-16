# 读取 ISIC 2020 的一个子集：train/ 下若干 jpg + train.csv
# 输出 (img1, img2, label)；label=1 表“同类”（正对/负对由你的定义确定）
import os, random, csv
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PairDataset(Dataset):
    def __init__(self, root: str, csv_name="train.csv", img_dir="train", image_size=224, mode="train", pair_ratio=1.0, seed=42):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.csv_path = os.path.join(root, csv_name)
        self.mode = mode
        self.rng = random.Random(seed)

        # 读取 csv：id,label（0/1）
        self.items = []  # (path, label)
        with open(self.csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("image_name") or row.get("image_id") or row.get("image") or row.get("isic_id")
                label = int(row.get("target") or row.get("label") or row.get("malignant", 0))
                img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
                if os.path.exists(img_path):
                    self.items.append((img_path, label))

        # 分桶便于采样正/负对
        self.pos_pool = [p for p in self.items if p[1] == 1]
        self.neg_pool = [p for p in self.items if p[1] == 0]
        assert len(self.pos_pool) > 0 and len(self.neg_pool) > 0, "需要正负样本"

        # 预处理
        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

        self.length = int(len(self.items) * pair_ratio)

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
