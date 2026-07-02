"""Load and access the YAML configuration."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict
import yaml


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get(cfg: Dict[str, Any], dotted: str, default=None):
    """Get a nested value with a dotted key, e.g. get(cfg, 'train.lr')."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
