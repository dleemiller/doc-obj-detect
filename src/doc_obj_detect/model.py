"""Model architecture for document object detection.

Combines ConvNeXt-DINOv3 backbone with D-FINE detection head.
"""

from typing import Any

from transformers import AutoImageProcessor, DFineConfig, DFineForObjectDetection


def create_model(
    backbone: str,
    num_classes: int,
    use_pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
    image_size: int = 512,
    **dfine_kwargs: Any,
) -> tuple[DFineForObjectDetection, AutoImageProcessor]:
    """Create D-FINE model with custom backbone.

    Args:
        backbone: Backbone model name (e.g., "convnext_large.dinov3_lvd1689m")
        num_classes: Number of object detection classes
        use_pretrained_backbone: Whether to use pretrained backbone weights
        freeze_backbone: Whether to freeze backbone parameters during training
        image_size: Input image size
        **dfine_kwargs: Additional D-FINE configuration

    Returns:
        Tuple of (model, image_processor)
    """
    # Extract backbone_kwargs for timm integration
    backbone_kwargs = dfine_kwargs.pop("backbone_kwargs", {"out_indices": (1, 2, 3)})

    # Configure D-FINE with timm backbone
    config = DFineConfig(
        backbone=backbone,
        use_timm_backbone=True,
        use_pretrained_backbone=use_pretrained_backbone,
        backbone_kwargs=backbone_kwargs,
        num_labels=num_classes,
        **dfine_kwargs,
    )

    # Initialize model from config
    model = DFineForObjectDetection(config)

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.model.backbone.parameters():
            param.requires_grad = False

    # Get image processor
    # Prefer the DFine preprocessor, fallback to Deformable DETR when unavailable
    processor_candidates = [
        "ustc-community/dfine-xlarge-obj2coco",
        "ustc-community/dfine-xlarge-coco",
    ]
    image_processor = None
    last_error: Exception | None = None

    for processor_id in processor_candidates:
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                processor_id,
                do_resize=False,  # Albumentations handles resize (including multi-scale)
                do_pad=True,  # Pad to make batch-compatible
                size={"height": image_size, "width": image_size},  # Used for padding reference
            )
            if processor_id != processor_candidates[0]:
                print(
                    f"[Model] Falling back to {processor_id} for image processing "
                    f"(preferred {processor_candidates[0]} unavailable)."
                )
            break
        except OSError as error:
            last_error = error

    if image_processor is None:
        raise (
            last_error if last_error is not None else RuntimeError("Failed to load image processor")
        )

    return model, image_processor


def get_trainable_parameters(model: DFineForObjectDetection) -> dict[str, Any]:
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
