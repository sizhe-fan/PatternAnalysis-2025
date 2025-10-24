# train.py
# 二元交叉熵 + AUC/ACC；与早先跑到 ~0.82–0.83 的设置一致（简洁、稳定）
import os, argparse, time
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
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)  # Win 上 0 最稳
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save", type=str, default="checkpoints/best.pt")
    return ap.parse_args()


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Data ===
    full_ds = PairDataset(args.data_root, image_size=args.image_size, mode="train")
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # === Model / Optim ===
    model = SiameseNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # 会有 FutureWarning，但不影响使用

    best_auc = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running = 0.0
        n_samples = 0

        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):  # 也会提示 FutureWarning，可先忽略
                logits = model(x1, x2)
                logits = logits.view(-1)    # 兼容 (B,1) 或 (B,)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            running += loss.item() * bs
            n_samples += bs

        train_loss = running / max(1, n_samples)

        # ---- val ----
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                logits = model(x1, x2)
                logits = logits.view(-1)
                all_logits.append(logits.cpu())
                all_labels.append(y)

        if len(all_logits) == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_acc=NA | val_auc=NA")
            continue

        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        metrics = step_binary_metrics(logits_cat, labels_cat)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_acc={metrics['acc']:.3f} | val_auc={metrics['auc']:.3f}")

        if metrics["auc"] == metrics["auc"] and metrics["auc"] > best_auc:  # 防 NaN
            best_auc = metrics["auc"]
            save_ckpt(model, args.save)
            print(f"  ✓ Saved best checkpoint to: {args.save} (AUC={best_auc:.3f})")

    mins = (time.time() - t0) / 60
    print(f"Total training time: {mins:.1f} min")


if __name__ == "__main__":
    main()
