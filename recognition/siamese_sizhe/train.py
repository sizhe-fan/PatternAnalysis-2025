# train.py
# Trains Siamese model and auto-saves: loss_curve.png, acc_curve.png, roc_curve.png
import os, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

from modules import SiameseNet
from dataset import PairDataset
from utils import step_binary_metrics, save_ckpt


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="dir containing train.csv and train/")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)         # 0 is safest on Windows
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save", type=str, default="checkpoints/best.pt")
    ap.add_argument("--plots_dir", type=str, default="images") # where to save figures
    return ap.parse_args()


@torch.no_grad()
def eval_logits(model, loader, device, loss_fn=None):
    """Return logits (tensor), labels (tensor), optional val_loss."""
    model.eval()
    all_logits, all_labels = [], []
    val_loss_total, n = 0.0, 0
    for x1, x2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)
        y = y.to(device)
        logits = model(x1, x2)
        logits = logits.view(-1)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
        if loss_fn is not None:
            val_loss_total += loss_fn(logits, y).item() * y.size(0)
            n += y.size(0)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    val_loss = (val_loss_total / n) if n > 0 and loss_fn is not None else None
    return logits_cat, labels_cat, val_loss


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Data ===
    full_ds = PairDataset(args.data_root, image_size=args.image_size, mode="train")
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # === Model / Optim ===
    model = SiameseNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # FutureWarning is OK

    # for curves
    hist_train_loss = []
    hist_val_loss   = []
    hist_val_acc    = []

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
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x1, x2)
                logits = logits.view(-1)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            running += loss.item() * bs
            n_samples += bs

        train_loss = running / max(1, n_samples)

        # ---- validation (get logits, labels, val_loss) ----
        logits_cat, labels_cat, val_loss = eval_logits(model, val_loader, device, loss_fn=loss_fn)
        metrics = step_binary_metrics(logits_cat, labels_cat)

        # log
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={(val_loss or 0):.4f} | "
              f"val_acc={metrics['acc']:.3f} | val_auc={metrics['auc']:.3f}")

        # curves history
        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss if val_loss is not None else float('nan'))
        hist_val_acc.append(metrics['acc'])

        # save best by AUC
        if metrics["auc"] == metrics["auc"] and metrics["auc"] > best_auc:  # guard NaN
            best_auc = metrics["auc"]
            save_ckpt(model, args.save)
            print(f"  ✓ Saved best checkpoint to: {args.save} (AUC={best_auc:.3f})")

    mins = (time.time() - t0) / 60
    print(f"Total training time: {mins:.1f} min")

    # === Plot 1: Loss curves ===
    plt.figure()
    plt.plot(hist_train_loss, label="Train Loss")
    plt.plot(hist_val_loss,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    loss_path = os.path.join(args.plots_dir, "loss_curve.png")
    plt.savefig(loss_path, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"Saved: {loss_path}")

    # === Plot 2: Accuracy curve ===
    plt.figure()
    plt.plot(hist_val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    acc_path = os.path.join(args.plots_dir, "acc_curve.png")
    plt.savefig(acc_path, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"Saved: {acc_path}")

    # === Plot 3: ROC curve (reload best checkpoint, recompute on val) ===
    #   This ensures the ROC is computed from the best-AUC model.
    best_model = SiameseNet().to(device)
    state = torch.load(args.save, map_location=device)
    best_model.load_state_dict(state)
    best_logits, best_labels, _ = eval_logits(best_model, val_loader, device, loss_fn=None)

    # sigmoid to get probabilities
    y_score = torch.sigmoid(best_logits).numpy()
    y_true  = best_labels.numpy()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Best Checkpoint)")
    plt.legend(loc="lower right")
    roc_path = os.path.join(args.plots_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"Saved: {roc_path}")


if __name__ == "__main__":
    main()
