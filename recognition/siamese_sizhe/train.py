# 训练脚本：二元交叉熵 + AUC/ACC；建议用小数据子集先跑通
import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from modules import SiameseNet
from dataset import PairDataset
from utils import step_binary_metrics, save_ckpt

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="包含 train.csv 与 train/ 的目录")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save", type=str, default="recognition/siamese_sizhe/checkpoints/best.pt")
    return ap.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_ds = PairDataset(args.data_root, image_size=args.image_size, mode="train")
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = SiameseNet(feat_dim=128, distance="l2").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_auc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x1, x2)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x1.size(0)
        train_loss = running / len(train_loader.dataset)

        # val
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                all_logits.append(logits)
                all_labels.append(y)
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        m = step_binary_metrics(logits, labels)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_acc={m['acc']:.3f} | val_auc={m['auc']:.3f}")

        if m["auc"] > best_auc:
            best_auc = m["auc"]
            save_ckpt(model, args.save)
            print(f"  ✓ Saved best checkpoint to: {args.save} (AUC={best_auc:.3f})")

if __name__ == "__main__":
    main()
