"""ModelFactory for building ConvNeXt+D-FINE detectors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoImageProcessor, DFineConfig, DFineForObjectDetection

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    model: DFineForObjectDetection
    processor: AutoImageProcessor


class ModelFactory:
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        *,
        use_pretrained_backbone: bool = True,
        use_pretrained_head: bool = True,
        freeze_backbone: bool = False,
        freeze_backbone_epochs: int | None = None,
        image_size: int = 512,
        pretrained_checkpoint: str | None = None,
        processor_candidates: tuple[str, ...] | None = None,
        id2label: dict[int, str] | None = None,
        **dfine_kwargs: Any,
    ) -> None:
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_pretrained_head = use_pretrained_head
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.image_size = image_size
        self.pretrained_checkpoint = pretrained_checkpoint
        self.processor_candidates = processor_candidates or (
            "ustc-community/dfine-xlarge-obj2coco",
            "ustc-community/dfine-xlarge-coco",
        )
        self.id2label = id2label
        self.dfine_kwargs = dfine_kwargs

    @classmethod
    def from_config(
        cls,
        model_cfg,
        dfine_cfg: dict[str, Any],
        image_size: int,
        id2label: dict[int, str] | None = None,
    ) -> ModelFactory:
        return cls(
            backbone=model_cfg.backbone,
            num_classes=model_cfg.num_classes,
            use_pretrained_backbone=model_cfg.use_pretrained_backbone,
            use_pretrained_head=model_cfg.use_pretrained_head,
            freeze_backbone=model_cfg.freeze_backbone,
            freeze_backbone_epochs=model_cfg.freeze_backbone_epochs,
            image_size=image_size,
            pretrained_checkpoint=model_cfg.pretrained_checkpoint,
            id2label=id2label,
            **dfine_cfg,
        )

    def build(self) -> ModelArtifacts:
        dfine_kwargs = dict(self.dfine_kwargs)
        backbone_kwargs = dfine_kwargs.pop("backbone_kwargs", {"out_indices": (1, 2, 3)})

        logger.info("Building model with config:")
        logger.info("  backbone: %s", self.backbone)
        logger.info("  num_classes: %s", self.num_classes)
        logger.info("  use_pretrained_backbone: %s", self.use_pretrained_backbone)
        logger.info("  use_pretrained_head: %s", self.use_pretrained_head)
        logger.info("  num_feature_levels: %s", dfine_kwargs.get("num_feature_levels"))
        logger.info("  encoder_in_channels: %s", dfine_kwargs.get("encoder_in_channels"))
        logger.info("  feat_strides: %s", dfine_kwargs.get("feat_strides"))
        logger.info("  backbone_kwargs: %s", backbone_kwargs)

        # Prepare label mappings
        config_kwargs = {
            "backbone": self.backbone,
            "use_timm_backbone": True,
            "use_pretrained_backbone": self.use_pretrained_backbone,
            "backbone_kwargs": backbone_kwargs,
            "num_labels": self.num_classes,
            **dfine_kwargs,
        }

        # Add label mappings if provided
        if self.id2label is not None:
            config_kwargs["id2label"] = self.id2label
            config_kwargs["label2id"] = {v: k for k, v in self.id2label.items()}
            logger.info("Setting label mappings: %s", self.id2label)

        dfine_config = DFineConfig(**config_kwargs)

        logger.info("DFineConfig created with:")
        logger.info("  num_feature_levels: %s", dfine_config.num_feature_levels)
        logger.info("  encoder_in_channels: %s", dfine_config.encoder_in_channels)

        model = DFineForObjectDetection(dfine_config)
        logger.info(
            "Model created. encoder_input_proj has %d levels", len(model.model.encoder_input_proj)
        )

        # Reinitialize D-FINE head if not using pretrained weights
        if not self.use_pretrained_head:
            logger.info("Training D-FINE head from scratch (use_pretrained_head=False)")
            # Reinitialize all non-backbone parameters
            for name, module in model.named_modules():
                if "backbone" not in name and hasattr(module, "reset_parameters"):
                    try:
                        module.reset_parameters()
                    except AttributeError:
                        pass  # Some modules don't have reset_parameters
            logger.info("D-FINE head reinitialized with random weights")

        if self.freeze_backbone:
            for param in model.model.backbone.parameters():
                param.requires_grad = False

        if self.pretrained_checkpoint:
            checkpoint_path = Path(self.pretrained_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            filtered_state_dict = {}
            skipped_keys = []
            model_state = model.state_dict()
            for key, value in state_dict.items():
                if key not in model_state:
                    continue
                if value.shape != model_state[key].shape:
                    skipped_keys.append(key)
                    continue
                filtered_state_dict[key] = value
            model.load_state_dict(filtered_state_dict, strict=False)
            if skipped_keys:
                logger.info(
                    "Skipped loading weights for keys due to shape mismatch: %s", skipped_keys
                )

        processor = self._load_processor()
        return ModelArtifacts(model=model, processor=processor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_processor(self) -> AutoImageProcessor:
        last_error: Exception | None = None
        for processor_id in self.processor_candidates:
            try:
                processor = AutoImageProcessor.from_pretrained(
                    processor_id,
                    do_resize=False,
                    do_pad=True,
                    size={"height": self.image_size, "width": self.image_size},
                )
                if processor_id != self.processor_candidates[0]:
                    logger.warning(
                        "Falling back to %s for image processing (preferred %s unavailable).",
                        processor_id,
                        self.processor_candidates[0],
                    )
                return processor
            except OSError as error:  # keep trying fallbacks
                last_error = error
                continue
        raise (
            last_error if last_error is not None else RuntimeError("Failed to load image processor")
        )


def create_model(
    *,
    backbone: str,
    num_classes: int,
    use_pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
    image_size: int = 512,
    pretrained_checkpoint: str | None = None,
    id2label: dict[int, str] | None = None,
    **dfine_kwargs: Any,
):
    """Backward-compatible wrapper used by tests/scripts."""

    factory = ModelFactory(
        backbone=backbone,
        num_classes=num_classes,
        use_pretrained_backbone=use_pretrained_backbone,
        freeze_backbone=freeze_backbone,
        image_size=image_size,
        pretrained_checkpoint=pretrained_checkpoint,
        id2label=id2label,
        **dfine_kwargs,
    )
    artifacts = factory.build()
    return artifacts.model, artifacts.processor
