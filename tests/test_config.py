"""Tests for configuration loading and integration."""

from pathlib import Path

import pytest

from doc_obj_detect.config import load_distill_config, load_train_config


def test_all_config_files_load_successfully():
    """Test that all config files are valid and loadable."""
    configs_dir = Path("configs")
    if not configs_dir.exists():
        pytest.skip("Configs directory not found")

    for config_file in configs_dir.glob("*.yaml"):
        if "distill" in config_file.name:
            config = load_distill_config(config_file)
            assert config.teacher is not None
            assert config.model is not None
        else:
            config = load_train_config(config_file)
            assert config.model is not None
            assert config.data is not None


def test_config_converts_to_function_kwargs():
    """Test that config objects convert properly to function kwargs."""
    config_path = Path("configs/pretrain_publaynet.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")

    config = load_train_config(config_path)

    # Should convert cleanly to kwargs
    detr_kwargs = config.detr.model_dump()
    training_kwargs = config.training.model_dump()

    assert isinstance(detr_kwargs, dict)
    assert isinstance(training_kwargs, dict)
    assert "num_queries" in detr_kwargs
    assert "num_train_epochs" in training_kwargs
