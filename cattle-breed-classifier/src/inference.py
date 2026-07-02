"""Predict the breed of a single image (or a folder of images).

Usage:
    python -m src.inference --image path/to/cow.jpg --topk 3
    python -m src.inference --image path/to/folder --topk 3
"""
from __future__ import annotations
import argparse
import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image

from .config import load_config
from .dataset import build_transforms
from .model import CattleBreedClassifier
from .utils import get_device

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = CattleBreedClassifier(ckpt["backbone"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, classes, ckpt["image_size"]


@torch.no_grad()
def predict(model, classes, tf, image_path, device, topk=3):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    k = min(topk, len(classes))
    conf, idx = probs.topk(k)
    return [(classes[i], float(c)) for c, i in zip(conf, idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    ckpt_path = args.checkpoint or os.path.join(
        cfg["paths"]["output_dir"], cfg["paths"]["checkpoint_name"])
    model, classes, image_size = load_model(ckpt_path, device)
    tf = build_transforms(image_size, train=False)

    if os.path.isdir(args.image):
        paths = [p for p in sorted(glob.glob(os.path.join(args.image, "*")))
                 if p.lower().endswith(IMG_EXT)]
    else:
        paths = [args.image]

    for p in paths:
        preds = predict(model, classes, tf, p, device, args.topk)
        pretty = ", ".join(f"{name} ({conf*100:.1f}%)" for name, conf in preds)
        print(f"{os.path.basename(p)}: {pretty}")


if __name__ == "__main__":
    main()
