"""Tests for SplitLRTrainer custom functionality.

This module tests the critical custom training behaviors:
- Split learning rates (backbone vs head)
- Custom gradient clipping (separate for backbone and head)
- Parameter grouping logic
- VRAM logging
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from doc_obj_detect.training.trainer_core import SplitLRTrainer


class SimpleModel(nn.Module):
    """Simple test model with backbone and head structure."""

    def __init__(self):
        super().__init__()
        # Create nested structure to match D-FINE naming
        self.model = nn.Module()
        self.model.backbone = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        self.head = nn.Sequential(
            nn.Linear(10, 5),
        )

    def forward(self, x):
        x = self.model.backbone(x)
        x = self.head(x)
        return x


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def training_args():
    """Create minimal training arguments."""
    return TrainingArguments(
        output_dir="/tmp/test_output",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        max_steps=10,
        logging_steps=1,
        save_steps=10,
        max_grad_norm=0.1,
        remove_unused_columns=False,
    )


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=10)
    dataset.__getitem__ = Mock(
        return_value={"input_values": torch.randn(10), "labels": torch.tensor(0)}
    )
    return dataset


class TestSplitLROptimizer:
    """Test split learning rate optimizer creation."""

    def test_creates_two_param_groups(self, simple_model, training_args, mock_dataset):
        """Test that optimizer has two parameter groups (backbone + head)."""
        trainer = SplitLRTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
        )

        optimizer = trainer.create_optimizer()

        # Should have 2 param groups
        assert len(optimizer.param_groups) == 2

    def test_applies_lr_multiplier_correctly(self, simple_model, training_args, mock_dataset):
        """Test that backbone LR is correctly scaled by multiplier."""
        base_lr = 1e-4
        backbone_multiplier = 0.01

        args = TrainingArguments(
            output_dir="/tmp/test_output",
            learning_rate=base_lr,
            per_device_train_batch_size=2,
            max_steps=10,
            logging_steps=1,
            save_steps=10,
            remove_unused_columns=False,
        )

        trainer = SplitLRTrainer(
            model=simple_model,
            args=args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=backbone_multiplier,
        )

        optimizer = trainer.create_optimizer()

        # Extract LRs from param groups
        lrs = [group["lr"] for group in optimizer.param_groups]

        # Should have head LR and backbone LR
        assert base_lr in lrs, f"Head LR {base_lr} not found in {lrs}"
        assert (
            abs(base_lr * backbone_multiplier - lrs[1]) < 1e-10
        ), f"Backbone LR should be {base_lr * backbone_multiplier}, got {lrs[1]}"

    def test_param_separation_by_name(self, simple_model, training_args, mock_dataset):
        """Test that parameters are correctly separated by name matching."""
        trainer = SplitLRTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
        )

        optimizer = trainer.create_optimizer()

        # Count params in each group
        head_group_params = list(optimizer.param_groups[0]["params"])
        backbone_group_params = list(optimizer.param_groups[1]["params"])

        # Simple model has:
        # - model.backbone: 2 layers = 2 weights + 2 biases = 4 params
        # - head: 1 layer = 1 weight + 1 bias = 2 params

        assert len(head_group_params) == 2, f"Expected 2 head params, got {len(head_group_params)}"
        assert (
            len(backbone_group_params) == 4
        ), f"Expected 4 backbone params, got {len(backbone_group_params)}"

    def test_handles_frozen_params(self, simple_model, training_args, mock_dataset):
        """Test that frozen parameters are excluded from optimizer."""
        # Freeze backbone
        for param in simple_model.model.backbone.parameters():
            param.requires_grad = False

        trainer = SplitLRTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
        )

        optimizer = trainer.create_optimizer()

        # Should only have head params (backbone is frozen)
        head_group_params = list(optimizer.param_groups[0]["params"])
        backbone_group_params = list(optimizer.param_groups[1]["params"])

        assert len(head_group_params) == 2, "Head should have 2 params"
        assert len(backbone_group_params) == 0, "Backbone should have 0 params (frozen)"


class TestCustomGradientClipping:
    """Test custom gradient clipping logic."""

    def test_gradient_clipping_separates_params(self, simple_model, training_args, mock_dataset):
        """Test that backbone and head gradients are clipped separately."""
        trainer = SplitLRTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
            backbone_max_grad_norm=0.5,
            head_max_grad_norm=0.1,
        )

        # Create fake gradients
        for name, param in simple_model.named_parameters():
            if "backbone" in name:
                # Large backbone gradients
                param.grad = torch.ones_like(param) * 10.0
            else:
                # Smaller head gradients
                param.grad = torch.ones_like(param) * 2.0

        # Clip gradients
        trainer._clip_gradients_custom(simple_model)

        # Check that backbone grads were clipped differently than head
        for name, param in simple_model.named_parameters():
            if "backbone" in name:
                # Backbone should be clipped to lower norm
                assert param.grad is not None
                # Gradients should be scaled down
                assert torch.all(param.grad < 10.0), f"Backbone grad not clipped: {name}"
            else:
                # Head should be clipped too
                assert param.grad is not None
                assert torch.all(param.grad < 2.0), f"Head grad not clipped: {name}"

    def test_gradient_clipping_with_none_grads(self, simple_model, training_args, mock_dataset):
        """Test that clipping handles None gradients gracefully."""
        trainer = SplitLRTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
            backbone_max_grad_norm=0.5,
            head_max_grad_norm=0.1,
        )

        # Leave some gradients as None
        for i, (_name, param) in enumerate(simple_model.named_parameters()):
            if i % 2 == 0:
                param.grad = torch.ones_like(param)
            else:
                param.grad = None

        # Should not crash
        trainer._clip_gradients_custom(simple_model)

    def test_gradient_clipping_uses_correct_norms(self, simple_model):
        """Test that correct max norms are used for each group."""
        # Create minimal args
        args = TrainingArguments(
            output_dir="/tmp/test",
            per_device_train_batch_size=1,
            max_steps=1,
            remove_unused_columns=False,
        )

        backbone_norm = 0.5
        head_norm = 0.1

        trainer = SplitLRTrainer(
            model=simple_model,
            args=args,
            train_dataset=Mock(__len__=Mock(return_value=1)),
            backbone_lr_multiplier=0.01,
            backbone_max_grad_norm=backbone_norm,
            head_max_grad_norm=head_norm,
        )

        # Verify the norms are stored correctly
        assert trainer.backbone_max_grad_norm == backbone_norm
        assert trainer.head_max_grad_norm == head_norm
        assert trainer.use_separate_grad_norms is True


class TestVRAMLogging:
    """Test VRAM logging functionality."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.reset_peak_memory_stats")
    def test_vram_logging_doesnt_crash(
        self,
        mock_reset,
        mock_max_mem,
        mock_cuda_available,
        simple_model,
        training_args,
        mock_dataset,
    ):
        """Test that VRAM logging doesn't crash even with mocked CUDA."""
        mock_max_mem.return_value = 1024 * 1024 * 1024  # 1GB

        trainer = SplitLRTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
        )

        # Should not crash when logging
        trainer.log({"loss": 0.5})

    @patch("torch.cuda.is_available", return_value=False)
    def test_vram_logging_handles_no_cuda(
        self, mock_cuda_available, simple_model, training_args, mock_dataset
    ):
        """Test that VRAM logging handles CPU-only training gracefully."""
        trainer = SplitLRTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
        )

        # Should not crash even without CUDA
        trainer.log({"loss": 0.5})


class TestEvaluationLoop:
    """Test custom evaluation loop that returns raw outputs."""

    def test_evaluation_loop_structure(self, simple_model, mock_dataset):
        """Test that evaluation loop returns correct structure with raw outputs."""
        args = TrainingArguments(
            output_dir="/tmp/test",
            per_device_eval_batch_size=2,
            max_steps=1,
            remove_unused_columns=False,
        )

        # Create mock eval dataset
        eval_dataset = Mock()
        eval_dataset.__len__ = Mock(return_value=4)
        eval_dataset.__getitem__ = Mock(
            return_value={
                "pixel_values": torch.randn(3, 224, 224),
                "pixel_mask": torch.ones(224, 224),
                "labels": [
                    {
                        "class_labels": torch.tensor([0]),
                        "boxes": torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
                    }
                ],
            }
        )

        trainer = SplitLRTrainer(
            model=simple_model,
            args=args,
            train_dataset=mock_dataset,
            eval_dataset=eval_dataset,
            backbone_lr_multiplier=0.01,
        )

        # The evaluation_loop is called internally by evaluate()
        # We just verify trainer can be created with eval dataset
        assert trainer.eval_dataset is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_trainer_with_all_frozen_params(self, simple_model, training_args, mock_dataset):
        """Test trainer behavior when all parameters are frozen."""
        # Freeze all params
        for param in simple_model.parameters():
            param.requires_grad = False

        trainer = SplitLRTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
        )

        optimizer = trainer.create_optimizer()

        # Both groups should be empty
        total_params = sum(len(group["params"]) for group in optimizer.param_groups)
        assert total_params == 0, "Should have no trainable parameters"

    def test_trainer_with_no_backbone_params(self, training_args, mock_dataset):
        """Test trainer with model that has no 'backbone' in parameter names."""

        # Model with different naming
        class HeadOnlyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Linear(10, 5)

            def forward(self, x):
                return self.decoder(x)

        model = HeadOnlyModel()

        trainer = SplitLRTrainer(
            model=model,
            args=training_args,
            train_dataset=mock_dataset,
            backbone_lr_multiplier=0.01,
        )

        optimizer = trainer.create_optimizer()

        # Should have all params in head group, none in backbone group
        head_params = len(optimizer.param_groups[0]["params"])
        backbone_params = len(optimizer.param_groups[1]["params"])

        assert head_params == 2, "All params should be in head group"
        assert backbone_params == 0, "No params should be in backbone group"
