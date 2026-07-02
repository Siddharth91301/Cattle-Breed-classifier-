"""Dataset + transforms for folder-per-breed image data.

Expected layout (config.data.data_dir):
    data_dir/
        Gir/            *.jpg
        Sahiwal/        *.jpg
        Tharparkar/     *.jpg
        ...

Or, if config.data.has_split is true:
    data_dir/
        train/<breed>/*.jpg
        val/<breed>/*.jpg
        test/<breed>/*.jpg   (optional)
"""
from __future__ import annotations
import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _stratified_indices(targets: List[int], val_frac: float, test_frac: float, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_idx += idx[:n_test].tolist()
        val_idx += idx[n_test:n_test + n_val].tolist()
        train_idx += idx[n_test + n_val:].tolist()
    return train_idx, val_idx, test_idx



def _align_to_master(ds, master_class_to_idx):
    """Rewrite an ImageFolder's targets so its labels match a master class map.

    Splits can contain a subset of the classes (e.g. a breed missing from
    `test/`). ImageFolder numbers classes per-folder, which would misalign
    labels across splits. This remaps every sample to the master index.
    """
    remap = {ds.class_to_idx[c]: master_class_to_idx[c] for c in ds.classes}
    ds.samples = [(path, remap[t]) for path, t in ds.samples]
    ds.targets = [remap[t] for t in ds.targets]
    ds.imgs = ds.samples
    return ds


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    d = cfg["data"]
    image_size = d["image_size"]
    bs = cfg["train"]["batch_size"]
    nw = d.get("num_workers", 2)
    seed = d.get("seed", 42)

    train_tf = build_transforms(image_size, train=True)
    eval_tf = build_transforms(image_size, train=False)

    if d.get("has_split", False):
        train_ds = datasets.ImageFolder(os.path.join(d["data_dir"], "train"), train_tf)
        classes = train_ds.classes
        master = train_ds.class_to_idx
        val_ds = _align_to_master(
            datasets.ImageFolder(os.path.join(d["data_dir"], "val"), eval_tf), master)
        test_dir = os.path.join(d["data_dir"], "test")
        if os.path.isdir(test_dir):
            test_ds = _align_to_master(
                datasets.ImageFolder(test_dir, eval_tf), master)
        else:
            test_ds = val_ds
    else:
        full = datasets.ImageFolder(d["data_dir"])
        classes = full.classes
        tr_idx, va_idx, te_idx = _stratified_indices(
            full.targets, d.get("val_split", 0.15), d.get("test_split", 0.10), seed
        )
        # Wrap the same base folder with different transforms via three datasets.
        base_train = datasets.ImageFolder(d["data_dir"], train_tf)
        base_eval = datasets.ImageFolder(d["data_dir"], eval_tf)
        train_ds = Subset(base_train, tr_idx)
        val_ds = Subset(base_eval, va_idx)
        test_ds = Subset(base_eval, te_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=pin)
    return train_loader, val_loader, test_loader, classes
