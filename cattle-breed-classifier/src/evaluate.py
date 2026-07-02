"""Evaluate a trained checkpoint on the test split and produce plots.

Usage:
    python -m src.evaluate --config config.yaml
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config
from .dataset import build_dataloaders
from .model import build_model
from .utils import get_device


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(targets.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def plot_history(history_path, out_path):
    if not os.path.exists(history_path):
        return
    with open(history_path) as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, h["train_loss"], label="train")
    ax[0].plot(epochs, h["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(epochs, h["train_acc"], label="train")
    ax[1].plot(epochs, h["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_confusion(y_true, y_pred, classes, out_path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.6),
                                    max(5, len(classes) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    out_dir = cfg["paths"]["output_dir"]

    ckpt_path = os.path.join(out_dir, cfg["paths"]["checkpoint_name"])
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]

    _, _, test_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg, len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])

    y_true, y_pred = collect_preds(model, test_loader, device)

    from sklearn.metrics import classification_report, accuracy_score
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, zero_division=0)
    print(f"Test accuracy: {acc:.4f}\n")
    print(report)

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n{report}\n")
    plot_history(os.path.join(out_dir, "history.json"),
                 os.path.join(out_dir, "training_curves.png"))
    plot_confusion(y_true, y_pred, classes,
                   os.path.join(out_dir, "confusion_matrix.png"))
    print(f"Saved report + plots to '{out_dir}/'.")


if __name__ == "__main__":
    main()
