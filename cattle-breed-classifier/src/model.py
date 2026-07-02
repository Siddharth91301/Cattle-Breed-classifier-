"""Transfer-learning model built on a timm backbone."""
from __future__ import annotations
import timm
import torch
import torch.nn as nn


class CattleBreedClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int,
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone_name = backbone
        # num_classes=0 -> timm returns pooled features (no classifier head)
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable


def build_model(cfg, num_classes: int) -> CattleBreedClassifier:
    m = cfg["model"]
    return CattleBreedClassifier(
        backbone=m["backbone"],
        num_classes=num_classes,
        pretrained=m.get("pretrained", True),
        dropout=m.get("dropout", 0.2),
    )
