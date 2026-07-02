"""Small shared helpers."""
from __future__ import annotations
import json
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_label_map(classes, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mapping = {i: c for i, c in enumerate(classes)}
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def load_label_map(path: str) -> dict:
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
