"""Tests for model architecture."""

import torch

from doc_obj_detect.model import create_model, get_trainable_parameters


def test_create_model_basic():
    """Test basic model creation."""
    model, processor = create_model(
        backbone="resnet50",  # Use simpler backbone for testing
        num_classes=5,
        use_pretrained_backbone=False,
        freeze_backbone=False,
    )

    assert model is not None
    assert processor is not None
    assert model.config.num_labels == 5


def test_create_model_frozen_backbone():
    """Test model creation with frozen backbone."""
    model, _ = create_model(
        backbone="resnet50",
        num_classes=5,
        use_pretrained_backbone=False,
        freeze_backbone=True,
    )

    # Check that backbone parameters are frozen
    backbone_params_frozen = all(not p.requires_grad for p in model.model.backbone.parameters())
    assert backbone_params_frozen


def test_create_model_with_detr_kwargs():
    """Test model creation with DETR kwargs."""
    model, _ = create_model(
        backbone="resnet50",
        num_classes=10,
        use_pretrained_backbone=False,
        num_queries=100,
        encoder_layers=3,
        decoder_layers=3,
    )

    assert model.config.num_queries == 100
    assert model.config.encoder_layers == 3
    assert model.config.decoder_layers == 3


def test_get_trainable_parameters():
    """Test parameter counting."""
    model, _ = create_model(
        backbone="resnet50",
        num_classes=5,
        use_pretrained_backbone=False,
        freeze_backbone=False,
    )

    param_info = get_trainable_parameters(model)

    assert "total" in param_info
    assert "trainable" in param_info
    assert "frozen" in param_info
    assert "trainable_percent" in param_info

    assert param_info["total"] > 0
    assert param_info["trainable"] > 0
    assert param_info["trainable_percent"] > 0


def test_get_trainable_parameters_frozen():
    """Test parameter counting with frozen backbone."""
    model, _ = create_model(
        backbone="resnet50",
        num_classes=5,
        use_pretrained_backbone=False,
        freeze_backbone=True,
    )

    param_info = get_trainable_parameters(model)

    assert param_info["frozen"] > 0
    assert param_info["trainable_percent"] < 100


def test_model_forward_pass():
    """Test model forward pass."""
    model, processor = create_model(
        backbone="resnet50",
        num_classes=5,
        use_pretrained_backbone=False,
    )

    # Create dummy input
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 512, 512)

    # Forward pass
    outputs = model(pixel_values=pixel_values)

    assert outputs.logits is not None
    assert outputs.pred_boxes is not None
    assert outputs.logits.shape[0] == batch_size
    assert outputs.pred_boxes.shape[0] == batch_size
