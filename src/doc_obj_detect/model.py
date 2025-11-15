"""Model architecture for document object detection.

Combines Vision Transformer backbone with Deformable DETR detection head.
"""

from typing import Any

from transformers import (
    AutoImageProcessor,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
)


def create_model(
    backbone: str,
    num_classes: int,
    use_pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
    image_size: int = 512,
    **detr_kwargs: Any,
) -> tuple[DeformableDetrForObjectDetection, AutoImageProcessor]:
    """Create Deformable DETR model with custom backbone.

    Args:
        backbone: Backbone model name (e.g., "timm/vit_pe_spatial_base_patch16_512.fb")
        num_classes: Number of object detection classes
        use_pretrained_backbone: Whether to use pretrained backbone weights
        freeze_backbone: Whether to freeze backbone parameters during training
        image_size: Input image size
        **detr_kwargs: Additional DETR configuration (num_queries, encoder_layers, etc.)

    Returns:
        Tuple of (model, image_processor)
    """
    # Configure Deformable DETR with custom backbone
    config = DeformableDetrConfig(
        backbone=backbone,
        use_timm_backbone=True,
        use_pretrained_backbone=use_pretrained_backbone,
        num_labels=num_classes,
        auxiliary_loss=True,  # Enable auxiliary decoding losses for training
        **detr_kwargs,
    )

    # Initialize model from config
    model = DeformableDetrForObjectDetection(config)

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.model.backbone.parameters():
            param.requires_grad = False

    # Get image processor - use base deformable-detr processor
    image_processor = AutoImageProcessor.from_pretrained(
        "SenseTime/deformable-detr",
        do_resize=True,
        size={"shortest_edge": image_size, "longest_edge": image_size * 2},
    )

    return model, image_processor


def get_trainable_parameters(model: DeformableDetrForObjectDetection) -> dict[str, Any]:
    """Get information about trainable parameters.

    Args:
        model: The model to analyze

    Returns:
        Dict with parameter counts and percentages
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
    }
