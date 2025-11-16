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
        **detr_kwargs,
    )

    # Initialize model from config
    model = DeformableDetrForObjectDetection(config)

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.model.backbone.parameters():
            param.requires_grad = False

    # Get image processor - start from base deformable-detr processor
    # For ViT backbones, we want fixed square inputs
    image_processor = AutoImageProcessor.from_pretrained(
        "SenseTime/deformable-detr",
        do_resize=True,
        do_pad=True,
        size={"height": image_size, "width": image_size},
    )

    # If we are using a *frozen* pretrained backbone that has its own processor,
    # align the normalization stats (image_mean / image_std) with that backbone.
    #
    # This mainly helps HF backbones like "google/vit-base-patch16-224-in21k".
    # For timm/* backbones there usually is no HF processor, and they are
    # typically ImageNet-pretrained with the same mean/std as deformable-detr,
    # so we just keep the defaults.
    if use_pretrained_backbone and freeze_backbone:
        try:
            # Only try this if the backbone is a HF model, not timm/*
            if not backbone.startswith("timm/"):
                backbone_proc = AutoImageProcessor.from_pretrained(backbone)
                if getattr(backbone_proc, "image_mean", None) is not None:
                    image_processor.image_mean = backbone_proc.image_mean
                if getattr(backbone_proc, "image_std", None) is not None:
                    image_processor.image_std = backbone_proc.image_std
        except Exception:
            # If anything goes wrong (e.g. no processor for this backbone),
            # just fall back to the deformable-detr defaults.
            pass

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
