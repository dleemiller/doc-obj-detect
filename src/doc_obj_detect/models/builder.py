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
        freeze_backbone: bool = False,
        image_size: int = 512,
        pretrained_checkpoint: str | None = None,
        processor_candidates: tuple[str, ...] | None = None,
        **dfine_kwargs: Any,
    ) -> None:
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_pretrained_backbone = use_pretrained_backbone
        self.freeze_backbone = freeze_backbone
        self.image_size = image_size
        self.pretrained_checkpoint = pretrained_checkpoint
        self.processor_candidates = processor_candidates or (
            "ustc-community/dfine-xlarge-obj2coco",
            "ustc-community/dfine-xlarge-coco",
        )
        self.dfine_kwargs = dfine_kwargs

    @classmethod
    def from_config(cls, model_cfg, dfine_cfg: dict[str, Any], image_size: int) -> ModelFactory:
        return cls(
            backbone=model_cfg.backbone,
            num_classes=model_cfg.num_classes,
            use_pretrained_backbone=model_cfg.use_pretrained_backbone,
            freeze_backbone=model_cfg.freeze_backbone,
            image_size=image_size,
            pretrained_checkpoint=model_cfg.pretrained_checkpoint,
            **dfine_cfg,
        )

    def build(self) -> ModelArtifacts:
        dfine_kwargs = dict(self.dfine_kwargs)
        backbone_kwargs = dfine_kwargs.pop("backbone_kwargs", {"out_indices": (1, 2, 3)})
        dfine_config = DFineConfig(
            backbone=self.backbone,
            use_timm_backbone=True,
            use_pretrained_backbone=self.use_pretrained_backbone,
            backbone_kwargs=backbone_kwargs,
            num_labels=self.num_classes,
            **dfine_kwargs,
        )

        model = DFineForObjectDetection(dfine_config)

        if self.freeze_backbone:
            for param in model.model.backbone.parameters():
                param.requires_grad = False

        if self.pretrained_checkpoint:
            checkpoint_path = Path(self.pretrained_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

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
        **dfine_kwargs,
    )
    artifacts = factory.build()
    return artifacts.model, artifacts.processor
