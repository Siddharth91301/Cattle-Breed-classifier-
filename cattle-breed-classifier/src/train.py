"""Training entry point.

Usage:
    python -m src.train --config config.yaml
"""
from __future__ import annotations
import argparse
import os
import json
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import load_config
from .dataset import build_dataloaders
from .model import build_model
from .utils import set_seed, get_device, save_label_map, AverageMeter


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    train = optimizer is not None
    model.train(train)
    loss_m, acc_m = AverageMeter(), AverageMeter()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, targets in tqdm(loader, leave=False):
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, targets)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            bs = imgs.size(0)
            loss_m.update(loss.item(), bs)
            acc_m.update(accuracy(logits, targets), bs)
    return loss_m.avg, acc_m.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"].get("seed", 42))
    device = get_device()
    print(f"Device: {device}")

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, classes = build_dataloaders(cfg)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    save_label_map(classes, os.path.join(out_dir, "label_map.json"))

    model = build_model(cfg, num_classes).to(device)

    t = cfg["train"]
    criterion = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                                  weight_decay=t.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])

    use_amp = bool(t.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    freeze_epochs = t.get("freeze_epochs", 0)
    if freeze_epochs > 0:
        model.set_backbone_trainable(False)
        print(f"Backbone frozen for first {freeze_epochs} epoch(s).")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val = 0.0
    patience = t.get("early_stopping_patience", 10)
    bad_epochs = 0

    for epoch in range(1, t["epochs"] + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.set_backbone_trainable(True)
            print("Backbone unfrozen — training end-to-end.")

        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scaler, use_amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device,
                                    use_amp=use_amp)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch:02d}/{t['epochs']} | "
              f"train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f} | "
              f"lr {lr_now:.2e} | {time.time()-start:.0f}s")

        if va_acc > best_val:
            best_val = va_acc
            bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "classes": classes,
                "backbone": cfg["model"]["backbone"],
                "image_size": cfg["data"]["image_size"],
                "val_acc": va_acc,
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(out_dir, cfg["paths"]["checkpoint_name"]))
            print(f"  ✓ saved new best (val_acc={va_acc:.3f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement in {patience}).")
                break

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val acc: {best_val:.3f}. Artifacts in '{out_dir}/'.")


if __name__ == "__main__":
    main()
