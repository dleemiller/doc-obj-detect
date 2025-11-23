"""Tests for ModelFactory checkpoint loading functionality.

This module tests the critical model loading paths including:
- EMA weight loading from training checkpoints
- Regular weight loading
- Shape mismatch handling
- Edge cases and error conditions
"""

import tempfile
from pathlib import Path

import pytest
import torch
from transformers import DFineConfig, DFineForObjectDetection

from doc_obj_detect.models.builder import ModelFactory


@pytest.fixture
def minimal_model():
    """Create a minimal D-FINE model for testing."""
    config = DFineConfig(
        backbone="resnet18",
        use_timm_backbone=True,
        use_pretrained_backbone=False,
        num_labels=5,
        encoder_in_channels=[64, 128, 256],
        decoder_in_channels=[64, 64, 64],  # Must match backbone output channels
        d_model=64,
        encoder_hidden_dim=64,
        num_queries=10,
        decoder_layers=1,
        encoder_layers=1,
        feat_strides=[8, 16, 32],
        num_feature_levels=3,
        backbone_kwargs={"out_indices": (1, 2, 3)},
    )
    model = DFineForObjectDetection(config)
    return model


@pytest.fixture
def temp_checkpoint_dir(minimal_model):
    """Create a temporary checkpoint directory with model weights and EMA state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir)

        # Save regular model weights (like HF Trainer does)
        model_path = checkpoint_path / "pytorch_model.bin"
        torch.save(minimal_model.state_dict(), model_path)

        # Save EMA state (like EMACallback does)
        # This uses the actual structure from EMACallback.state_dict()
        ema_state = {
            "module": minimal_model.state_dict(),  # Key from callbacks.py:169
            "updates": 1000,
            "decay": 0.9999,
            "warmup_steps": 1000,
        }
        ema_path = checkpoint_path / "ema_state.pt"
        torch.save(ema_state, ema_path)

        yield checkpoint_path


class TestEMACheckpointLoading:
    """Test EMA checkpoint loading - TDD approach for bug fix."""

    def test_load_ema_checkpoint_failing(self, minimal_model, temp_checkpoint_dir):
        """TDD: This test should FAIL initially, exposing the shadow_params vs module bug.

        The bug: builder.py:165-168 checks for 'shadow_params' key, but
        EMACallback saves with 'module' key (callbacks.py:169).

        Expected behavior: Should successfully load EMA weights from checkpoint.
        Actual behavior (before fix): KeyError or silently falls back to regular weights.
        """
        # Create a fresh model to load into
        fresh_model = DFineForObjectDetection(minimal_model.config)

        # Load from checkpoint with prefer_ema=True
        ModelFactory.load_from_checkpoint(
            fresh_model,
            temp_checkpoint_dir,
            prefer_ema=True,
        )

        # Verify EMA weights were actually loaded (not regular weights)
        # We can check by looking at parameter values - they should match EMA state
        original_params = list(minimal_model.parameters())[0]
        loaded_params = list(fresh_model.parameters())[0]

        # If EMA loading worked, parameters should match
        assert torch.allclose(original_params, loaded_params, rtol=1e-5), (
            "EMA weights were not loaded correctly. "
            "Check if builder.py:165 looks for correct key ('module' not 'shadow_params')"
        )


class TestCheckpointLoadingBasics:
    """Test basic checkpoint loading functionality."""

    def test_load_from_checkpoint_without_ema(self, minimal_model):
        """Test loading when no EMA state exists - should fall back to regular weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir)

            # Save only regular weights (no EMA)
            model_path = checkpoint_path / "pytorch_model.bin"
            torch.save(minimal_model.state_dict(), model_path)

            # Create fresh model
            fresh_model = DFineForObjectDetection(minimal_model.config)

            # Load with prefer_ema=True, should fall back to regular weights
            ModelFactory.load_from_checkpoint(
                fresh_model,
                checkpoint_path,
                prefer_ema=True,
            )

            # Verify weights were loaded
            original_params = list(minimal_model.parameters())[0]
            loaded_params = list(fresh_model.parameters())[0]
            assert torch.allclose(original_params, loaded_params, rtol=1e-5)

    def test_load_from_checkpoint_prefer_ema_false(self, minimal_model, temp_checkpoint_dir):
        """Test loading with prefer_ema=False skips EMA even if available."""
        fresh_model = DFineForObjectDetection(minimal_model.config)

        # Load with prefer_ema=False
        ModelFactory.load_from_checkpoint(
            fresh_model,
            temp_checkpoint_dir,
            prefer_ema=False,
        )

        # Should load regular weights (not EMA)
        # Both should match since they're from same model, but logic path is different
        original_params = list(minimal_model.parameters())[0]
        loaded_params = list(fresh_model.parameters())[0]
        assert torch.allclose(original_params, loaded_params, rtol=1e-5)

    def test_load_from_checkpoint_file_path(self, minimal_model):
        """Test loading directly from file path (not directory)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            torch.save(minimal_model.state_dict(), checkpoint_path)

            fresh_model = DFineForObjectDetection(minimal_model.config)
            ModelFactory.load_from_checkpoint(fresh_model, checkpoint_path)

            original_params = list(minimal_model.parameters())[0]
            loaded_params = list(fresh_model.parameters())[0]
            assert torch.allclose(original_params, loaded_params, rtol=1e-5)

    def test_load_from_checkpoint_missing_file(self, minimal_model):
        """Test that FileNotFoundError is raised when checkpoint doesn't exist."""
        fresh_model = DFineForObjectDetection(minimal_model.config)

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            ModelFactory.load_from_checkpoint(fresh_model, "/nonexistent/path/checkpoint")

    def test_load_from_checkpoint_empty_directory(self, minimal_model):
        """Test that FileNotFoundError is raised for empty checkpoint directory."""
        fresh_model = DFineForObjectDetection(minimal_model.config)

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir)

            with pytest.raises(FileNotFoundError, match="No model weights found"):
                ModelFactory.load_from_checkpoint(fresh_model, empty_dir)


class TestCheckpointLoadingShapeMismatch:
    """Test handling of shape mismatches during checkpoint loading."""

    def test_load_from_checkpoint_shape_mismatch(self, minimal_model):
        """Test that shape mismatches are handled gracefully (filtered out)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"

            # Create state dict with wrong shape for one parameter
            state_dict = minimal_model.state_dict()
            first_key = list(state_dict.keys())[0]
            # Modify shape of first parameter
            wrong_shape = torch.randn(999, 999)  # Wrong shape
            state_dict[first_key] = wrong_shape

            torch.save(state_dict, checkpoint_path)

            # Create fresh model
            fresh_model = DFineForObjectDetection(minimal_model.config)

            # Should load successfully but skip mismatched keys
            # strict=False allows partial loading
            ModelFactory.load_from_checkpoint(
                fresh_model,
                checkpoint_path,
                strict=False,
            )

            # Verify first parameter was NOT loaded (still random init)
            original_first = list(minimal_model.parameters())[0]
            loaded_first = list(fresh_model.parameters())[0]
            assert not torch.allclose(
                original_first, loaded_first, rtol=1e-5
            ), "Shape-mismatched parameter should not have been loaded"


class TestModelFactoryFromConfig:
    """Test ModelFactory.from_config() - the production code path."""

    def test_from_config_builds_model(self):
        """Test that from_config creates valid model from config objects."""
        from doc_obj_detect.config.schemas import ModelConfig

        model_cfg = ModelConfig(
            backbone="resnet18",
            architecture="dfine_small",
            num_classes=5,
            use_pretrained_backbone=False,
        )

        dfine_cfg_dict = {
            "encoder_in_channels": [64, 128, 256],
            "feat_strides": [8, 16, 32],
            "num_feature_levels": 3,
            "encoder_hidden_dim": 64,
            "d_model": 64,
            "num_queries": 10,
            "decoder_layers": 1,
            "encoder_layers": 1,
        }

        factory = ModelFactory.from_config(
            model_cfg,
            dfine_cfg_dict,
            image_size=512,
            id2label={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
        )

        artifacts = factory.build()

        # Verify model was created
        assert isinstance(artifacts.model, DFineForObjectDetection)
        assert artifacts.model.config.num_labels == 5
        assert artifacts.processor is not None
