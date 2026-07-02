"""Gradio web demo: upload a cattle image, get the predicted breed.

Usage:
    python app/app.py                 # uses outputs/best_model.pt
    python app/app.py --checkpoint path/to/best_model.pt
"""
from __future__ import annotations
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import build_transforms          # noqa: E402
from src.model import CattleBreedClassifier        # noqa: E402
from src.utils import get_device                   # noqa: E402


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = CattleBreedClassifier(ckpt["backbone"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tf = build_transforms(ckpt["image_size"], train=False)
    return model, classes, tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join("outputs", "best_model.pt"))
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = ap.parse_args()

    device = get_device()
    model, classes, tf = load(args.checkpoint, device)

    @torch.no_grad()
    def classify(img: Image.Image):
        if img is None:
            return {}
        x = tf(img.convert("RGB")).unsqueeze(0).to(device)
        probs = F.softmax(model(x), dim=1)[0].cpu()
        return {classes[i]: float(probs[i]) for i in range(len(classes))}

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Cattle image"),
        outputs=gr.Label(num_top_classes=5, label="Predicted breed"),
        title="Indian Cattle Breed Classifier",
        description="Upload a photo of a cow/bull to identify its breed.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
