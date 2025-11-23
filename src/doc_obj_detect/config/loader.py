"""Helpers for loading YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import ArchitectureConfig, DistillConfig, TrainConfig


def load_architecture_config(architecture_name: str, config_root: Path | None = None) -> dict:
    """Load architecture configuration from configs/architectures/*.yaml.

    Args:
        architecture_name: Name of the architecture (e.g., 'dfine_xlarge')
        config_root: Root directory containing configs/ (defaults to project root)

    Returns:
        Dictionary containing architecture parameters
    """
    if config_root is None:
        # Default to project root (3 levels up from this file)
        config_root = Path(__file__).parent.parent.parent.parent

    arch_path = config_root / "configs" / "architectures" / f"{architecture_name}.yaml"

    if not arch_path.exists():
        raise FileNotFoundError(
            f"Architecture config not found: {arch_path}\n"
            f"Available architectures: {list((config_root / 'configs' / 'architectures').glob('*.yaml'))}"
        )

    with open(arch_path) as f:
        arch_dict = yaml.safe_load(f)

    # Validate architecture config
    ArchitectureConfig(**arch_dict)
    return arch_dict


def load_train_config(config_path: str | Path) -> TrainConfig:
    """Load training configuration and merge with architecture config.

    This function:
    1. Loads the training config YAML
    2. Extracts the architecture name from model.architecture
    3. Loads the corresponding architecture config
    4. Merges architecture params into dfine section
    5. Returns validated TrainConfig

    Args:
        config_path: Path to training config YAML

    Returns:
        Validated TrainConfig with merged architecture parameters
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Extract architecture name
    architecture_name = config_dict.get("model", {}).get("architecture")
    if not architecture_name:
        raise ValueError("Training config must specify 'model.architecture' (e.g., 'dfine_xlarge')")

    # Load architecture config
    config_root = (
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    )
    arch_dict = load_architecture_config(architecture_name, config_root)

    # Merge architecture into dfine section
    # Training config params override architecture defaults
    if "dfine" not in config_dict:
        config_dict["dfine"] = {}

    # Architecture params are defaults; training config overrides
    merged_dfine = {**arch_dict, **config_dict["dfine"]}
    config_dict["dfine"] = merged_dfine

    return TrainConfig(**config_dict)


def load_distill_config(config_path: str | Path) -> DistillConfig:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return DistillConfig(**config_dict)
