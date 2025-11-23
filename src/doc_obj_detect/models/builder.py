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
    """Factory for building D-FINE object detection models with custom backbones.

    Builds models from scratch with pretrained backbone and random D-FINE head.
    For loading from checkpoints, use load_from_checkpoint() static method.

    Checkpoint loading via CLI flags:
    - --resume: Full state resume (handled by HF Trainer)
    - --load: Weights only (handled by load_from_checkpoint method)
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        *,
        use_pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        freeze_backbone_epochs: int | None = None,
        image_size: int = 512,
        id2label: dict[int, str] | None = None,
        **dfine_kwargs: Any,
    ) -> None:
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_pretrained_backbone = use_pretrained_backbone
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.image_size = image_size
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
            freeze_backbone=model_cfg.freeze_backbone,
            freeze_backbone_epochs=model_cfg.freeze_backbone_epochs,
            image_size=image_size,
            id2label=id2label,
            **dfine_cfg,
        )

    def build(self) -> ModelArtifacts:
        """Build model from scratch with random D-FINE head.

        Creates new model with pretrained backbone (if use_pretrained_backbone=True)
        and randomly initialized D-FINE head.

        To load from checkpoint, use load_from_checkpoint() after building.
        """
        dfine_kwargs = dict(self.dfine_kwargs)
        backbone_kwargs = dfine_kwargs.pop("backbone_kwargs", {"out_indices": (1, 2, 3)})

        logger.info(
            "Building D-FINE model: backbone=%s, num_classes=%d, pretrained_backbone=%s, levels=%d",
            self.backbone,
            self.num_classes,
            self.use_pretrained_backbone,
            dfine_kwargs.get("num_feature_levels", 3),
        )

        model = self._build_from_scratch(backbone_kwargs, dfine_kwargs)

        if self.freeze_backbone:
            for param in model.model.backbone.parameters():
                param.requires_grad = False

        processor = self._load_processor()
        return ModelArtifacts(model=model, processor=processor)

    def _build_from_scratch(
        self, backbone_kwargs: dict[str, Any], dfine_kwargs: dict[str, Any]
    ) -> DFineForObjectDetection:
        """Build model from scratch with random D-FINE head initialization."""
        # Prepare config (DFineConfig will validate all parameters)
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

        # Create config and model
        dfine_config = DFineConfig(**config_kwargs)
        model = DFineForObjectDetection(dfine_config)

        logger.info(
            "Model created: encoder_hidden_dim=%d, decoder_layers=%d, d_model=%d",
            dfine_config.encoder_hidden_dim,
            dfine_config.decoder_layers,
            dfine_config.d_model,
        )

        return model

    @staticmethod
    def load_from_checkpoint(
        model: DFineForObjectDetection,
        checkpoint_path: str | Path,
        strict: bool = False,
        prefer_ema: bool = True,
    ) -> None:
        """Load model weights from a checkpoint (for --load flag).

        This method loads ONLY model weights (no optimizer state, no trainer state).
        Use this when you want to start new training with pretrained weights.

        Args:
            model: The model to load weights into
            checkpoint_path: Path to checkpoint directory or .pt/.bin file
            strict: Whether to require exact key matching (default: False)
            prefer_ema: If True and EMA weights exist, load those instead (default: True)

        Raises:
            FileNotFoundError: If checkpoint not found
        """
        checkpoint_path = Path(checkpoint_path)

        # Handle both checkpoint directories and direct file paths
        if checkpoint_path.is_dir():
            # Check for EMA weights first if preferred
            if prefer_ema:
                ema_path = checkpoint_path / "ema_state.pt"
                if ema_path.exists():
                    logger.info("Loading EMA weights from: %s", ema_path)
                    ema_state = torch.load(ema_path, map_location="cpu")
                    # EMA state dict can have different keys depending on implementation
                    # EMACallback saves with 'module' key (callbacks.py:169)
                    # Some implementations use 'shadow_params'
                    if "module" in ema_state:
                        state_dict = ema_state["module"]
                        logger.info("Found EMA weights under 'module' key")
                    elif "shadow_params" in ema_state:
                        state_dict = ema_state["shadow_params"]
                        logger.info("Found EMA weights under 'shadow_params' key")
                    else:
                        # Assume the entire state dict is the model weights
                        state_dict = ema_state
                        logger.info("Using entire EMA state as model weights")

                    filtered_state_dict = {}
                    skipped_keys = []
                    model_state = model.state_dict()

                    for key, value in state_dict.items():
                        if key not in model_state:
                            skipped_keys.append(f"{key} (not in model)")
                            continue
                        if value.shape != model_state[key].shape:
                            skipped_keys.append(
                                f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})"
                            )
                            continue
                        filtered_state_dict[key] = value

                    model.load_state_dict(filtered_state_dict, strict=strict)
                    logger.info(
                        "Loaded %d/%d EMA keys from checkpoint (%d skipped)",
                        len(filtered_state_dict),
                        len(state_dict),
                        len(skipped_keys),
                    )
                    if skipped_keys and len(skipped_keys) <= 5:
                        for key_msg in skipped_keys:
                            logger.info("  Skipped: %s", key_msg)
                    return

            # Fall back to regular model weights
            candidates = [
                checkpoint_path / "model.safetensors",
                checkpoint_path / "pytorch_model.bin",
                checkpoint_path / "model.pt",
            ]
            checkpoint_file = None
            for candidate in candidates:
                if candidate.exists():
                    checkpoint_file = candidate
                    break

            if checkpoint_file is None:
                raise FileNotFoundError(
                    f"No model weights found in checkpoint directory: {checkpoint_path}\n"
                    f"Looked for: {[c.name for c in candidates]}"
                )
            checkpoint_path = checkpoint_file

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info("Loading weights from checkpoint: %s", checkpoint_path)

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        filtered_state_dict = {}
        skipped_keys = []
        model_state = model.state_dict()

        for key, value in state_dict.items():
            if key not in model_state:
                skipped_keys.append(f"{key} (not in model)")
                continue
            if value.shape != model_state[key].shape:
                skipped_keys.append(
                    f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})"
                )
                continue
            filtered_state_dict[key] = value

        model.load_state_dict(filtered_state_dict, strict=strict)

        logger.info(
            "Loaded %d/%d keys from checkpoint (%d skipped)",
            len(filtered_state_dict),
            len(state_dict),
            len(skipped_keys),
        )

        if skipped_keys and len(skipped_keys) <= 5:
            for key_msg in skipped_keys:
                logger.info("  Skipped: %s", key_msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_processor(self) -> AutoImageProcessor:
        """Load image processor with standard ImageNet normalization for pretrained backbones."""
        return AutoImageProcessor.from_pretrained(
            "ustc-community/dfine-xlarge-obj365",  # Standard D-FINE processor
            do_resize=False,
            do_pad=True,
            do_normalize=True,  # CRITICAL: Enable normalization for pretrained backbone
            image_mean=[0.485, 0.456, 0.406],  # ImageNet mean (DINOv3 standard)
            image_std=[0.229, 0.224, 0.225],  # ImageNet std (DINOv3 standard)
            size={"height": self.image_size, "width": self.image_size},
        )


def create_model(
    *,
    backbone: str,
    num_classes: int,
    use_pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
    image_size: int = 512,
    id2label: dict[int, str] | None = None,
    **dfine_kwargs: Any,
):
    """Backward-compatible wrapper used by tests/scripts.

    To load from checkpoint, use ModelFactory.load_from_checkpoint() after creation.
    """
    factory = ModelFactory(
        backbone=backbone,
        num_classes=num_classes,
        use_pretrained_backbone=use_pretrained_backbone,
        freeze_backbone=freeze_backbone,
        image_size=image_size,
        id2label=id2label,
        **dfine_kwargs,
    )
    artifacts = factory.build()
    return artifacts.model, artifacts.processor
