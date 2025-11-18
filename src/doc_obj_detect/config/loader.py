"""Helpers for loading YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import DistillConfig, TrainConfig


def load_train_config(config_path: str | Path) -> TrainConfig:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return TrainConfig(**config_dict)


def load_distill_config(config_path: str | Path) -> DistillConfig:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return DistillConfig(**config_dict)
