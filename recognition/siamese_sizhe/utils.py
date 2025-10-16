import torch, os
from typing import Dict
from sklearn.metrics import roc_auc_score, accuracy_score

def step_binary_metrics(logits, labels, threshold=0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    pred = (probs >= threshold).astype("int32")
    return {
        "acc": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, probs)) if len(set(y.tolist())) > 1 else 0.0
    }

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
