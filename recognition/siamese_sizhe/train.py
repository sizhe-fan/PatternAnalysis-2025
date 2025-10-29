# train.py
"""
Training script for a Siamese Network on ISIC data.

- Uses BCEWithLogitsLoss for binary similarity prediction.
- Reports Accuracy and AUC per epoch.
- Saves the best checkpoint by validation AUC.
- Matches the configuration that previously reached ~0.82–0.83 AUC.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from modules import SiameseNet
from dataset import PairDataset
from utils import step_binary_metrics, save_ckpt


def get_args():
    """
    Parse command-line arguments.

    --data_root: folder containing train.csv and train/ images
    --image_size: resize side length for images
    --batch_size: batch size
    --workers: DataLoader workers (Windows users: 0 is safest)
    --epochs: number of training epochs
    --lr: learning rate
    --amp: enable mixed precision (automatic mixed precision)
    --save: path to save the best checkpoint
    """
    ap = argparse.ArgumentParser(description="Train a Siamese Network on ISIC pairs.")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Directory that contains train.csv and train/ folder.")
    ap.add_argument("--image_size", type=int, default=224,
                    help="Square resize for input images.")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Batch size.")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (use 0 on Windows for stability).")
    ap.add_argument("--epochs", type=int, default=30,
                    help="Number of training epochs.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate for AdamW.")
    ap.add_argument("--amp", action="store_true",
                    help="Enable mixed precision (AMP).")
    ap.add_argument("--save", type=str, default="checkpoints/best.pt",
                    help="Path to save best checkpoint (by AUC).")
    return ap.parse_args()


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Data =====
    # Build full dataset, then split (80% train / 20% val).
    full_ds = PairDataset(args.data_root, image_size=args.image_size, mode="train")
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # ===== Model / Optimizer / Loss =====
    model = SiameseNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # NOTE: torch.cuda.amp.GradScaler is deprecated in favor of torch.amp.GradScaler('cuda').
    # We keep this for compatibility; the FutureWarning is harmless.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_auc = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ----- Train -----
        model.train()
        running = 0.0
        n_samples = 0

        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            # NOTE: torch.cuda.amp.autocast is deprecated; warning can be ignored here.
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x1, x2)     # (B,) or (B,1)
                logits = logits.view(-1)   # ensure shape (B,)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            running += loss.item() * bs
            n_samples += bs

        train_loss = running / max(1, n_samples)

        # ----- Validation -----
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                logits = model(x1, x2).view(-1)  # (B,)
                all_logits.append(logits.cpu())
                all_labels.append(y)

        if len(all_logits) == 0:
            # Defensive path for empty validation (should not happen with a proper split)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_acc=NA | val_auc=NA")
            continue

        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        metrics = step_binary_metrics(logits_cat, labels_cat)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | "
            f"val_acc={metrics['acc']:.3f} | val_auc={metrics['auc']:.3f}"
        )

        # Save best checkpoint by AUC (ignore NaN with self-check)
        if metrics["auc"] == metrics["auc"] and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            save_ckpt(model, args.save)
            print(f"  ✓ Saved best checkpoint to: {args.save} (AUC={best_auc:.3f})")

    mins = (time.time() - t0) / 60
    print(f"Total training time: {mins:.1f} min")


if __name__ == "__main__":
    main()

