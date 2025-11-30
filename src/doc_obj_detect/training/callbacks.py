"""Custom Trainer callbacks used by the training runner."""

import logging
import math
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class UnfreezeBackboneCallback(TrainerCallback):
    """Unfreeze the backbone after a warmup period.

    HuggingFace's Trainer API does not provide a built-in mechanism to partially
    unfreeze parameters after a number of steps. This callback watches the
    training state and flips ``requires_grad`` for the backbone once the desired
    step or epoch is reached.

    Args:
        unfreeze_at_step: Unfreeze at this specific step (takes precedence)
        unfreeze_at_epoch: Unfreeze after this many epochs (calculated dynamically)
    """

    def __init__(
        self, unfreeze_at_step: int | None = None, unfreeze_at_epoch: int | None = None
    ) -> None:
        if unfreeze_at_step is None and unfreeze_at_epoch is None:
            raise ValueError("Must specify either unfreeze_at_step or unfreeze_at_epoch")
        self.unfreeze_at_step = unfreeze_at_step
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self._done = False
        self._target_step: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        """Calculate target step from epochs if needed."""
        if self.unfreeze_at_epoch is not None and self._target_step is None:
            # Calculate steps per epoch
            num_train_samples = (
                state.max_steps
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                / args.num_train_epochs
            )
            steps_per_epoch = num_train_samples / (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            self._target_step = int(self.unfreeze_at_epoch * steps_per_epoch)
            logger.info(
                "[Callback] Will unfreeze backbone after %d epochs (step %d, ~%.1f steps/epoch)",
                self.unfreeze_at_epoch,
                self._target_step,
                steps_per_epoch,
            )
        elif self.unfreeze_at_step is not None:
            self._target_step = self.unfreeze_at_step
            logger.info("[Callback] Will unfreeze backbone at step %d", self._target_step)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if self._done or self._target_step is None:
            return control

        if state.global_step >= self._target_step:
            model = kwargs.get("model")
            if model is None:
                return control

            backbone = getattr(model, "model", None)
            backbone = getattr(backbone, "backbone", None)
            if backbone is None:
                return control

            for param in backbone.parameters():
                param.requires_grad = True

            logger.info(
                "[Callback] Unfroze backbone at step %d (epoch %.2f)",
                state.global_step,
                state.epoch if hasattr(state, "epoch") else 0,
            )
            self._done = True
        return control


class ModelEMA:
    """Exponential Moving Average for model weights.

    Maintains a smoothed copy of model weights that typically generalizes better
    than the training weights. Based on D-FINE's EMA implementation.

    Args:
        model: Model to create EMA for
        decay: EMA decay rate (default: 0.9999)
        warmup_steps: Number of steps to warmly increase decay from 0 to target (default: 1000)
        device: Device to place EMA model on
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 1000,
        device: torch.device | str | None = None,
    ):
        # Create EMA model as a deep copy
        self.module = deepcopy(self._de_parallel(model))
        self.module.eval()

        # Disable gradients for EMA parameters
        for param in self.module.parameters():
            param.requires_grad_(False)

        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

        # Move to device if specified
        if device is not None:
            self.module.to(device)

        logger.info(
            "[EMA] Initialized with decay=%.4f, warmup_steps=%d",
            decay,
            warmup_steps,
        )

    @staticmethod
    def _de_parallel(model: nn.Module) -> nn.Module:
        """Unwrap DDP/FSDP model."""
        return model.module if hasattr(model, "module") else model

    def _get_decay(self) -> float:
        """Get current decay value with warmup schedule."""
        if self.warmup_steps == 0:
            return self.decay
        # Exponential warmup: decay increases from 0 to target over warmup_steps
        return self.decay * (1.0 - math.exp(-self.updates / self.warmup_steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights. Call after every optimizer step."""
        self.updates += 1
        decay = self._get_decay()

        # Get current model state
        model_state = self._de_parallel(model).state_dict()

        # Update EMA parameters
        for key, ema_param in self.module.state_dict().items():
            if ema_param.dtype.is_floating_point:
                # EMA formula: ema_param = decay * ema_param + (1 - decay) * model_param
                ema_param.mul_(decay).add_(model_state[key].detach(), alpha=1.0 - decay)

        # Log occasionally during warmup
        if self.updates <= self.warmup_steps and self.updates % 100 == 0:
            logger.info(
                "[EMA] Update %d/%d, effective_decay=%.6f",
                self.updates,
                self.warmup_steps,
                decay,
            )

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        # Get full state dict
        full_state_dict = self.module.state_dict()

        # Remove duplicate keys that share memory (D-FINE creates aliases for decoder heads)
        # Keep unprefixed keys (class_embed.*, bbox_embed.*) and remove prefixed duplicates
        deduplicated = self._deduplicate_state_dict(full_state_dict)

        return {
            "module": deduplicated,
            "updates": self.updates,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }

    @staticmethod
    def _deduplicate_state_dict(state_dict: dict) -> dict:
        """Remove duplicate keys that share memory.

        D-FINE models sometimes have both 'model.decoder.X' and 'X' keys pointing to same tensor.
        This happens because decoder heads (class_embed, bbox_embed) are aliased at module level.
        We keep the unprefixed keys and remove the prefixed duplicates to avoid issues with
        safetensors format, which cannot handle shared memory tensors.
        """
        # Build a mapping of data_ptr -> list of keys that share that memory
        ptr_to_keys: dict[int, list[str]] = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                ptr = tensor.data_ptr()
                if ptr not in ptr_to_keys:
                    ptr_to_keys[ptr] = []
                ptr_to_keys[ptr].append(key)

        # Identify which keys to keep
        keys_to_remove = set()
        for ptr, keys in ptr_to_keys.items():
            if len(keys) > 1:
                # Multiple keys share memory - keep the shortest/cleanest one
                # Prefer unprefixed keys (e.g., 'class_embed.*') over prefixed ('model.decoder.class_embed.*')
                keys_sorted = sorted(keys, key=lambda k: (len(k), k))
                # Keep first (shortest), remove the rest
                keys_to_remove.update(keys_sorted[1:])

        # Create deduplicated state dict
        deduplicated = {k: v for k, v in state_dict.items() if k not in keys_to_remove}

        if keys_to_remove:
            logger.info(
                "[EMA] Removed %d duplicate keys (shared memory) from state dict",
                len(keys_to_remove),
            )

        return deduplicated

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint.

        Uses strict=False to handle deduplicated state dicts that may be missing
        alias keys (e.g., 'model.decoder.class_embed.*' when 'class_embed.*' exists).
        """
        # Load with strict=False to allow missing alias keys
        # The deduplicated state may only have shorter keys (class_embed.*)
        # but the model has both short and long keys (model.decoder.class_embed.*) as aliases
        missing_keys, unexpected_keys = self.module.load_state_dict(state_dict["module"], strict=False)

        # Filter out expected missing keys (decoder head aliases)
        # These are prefixed versions of keys that exist in unprefixed form
        unexpected_missing = []
        for key in missing_keys:
            # Check if this is a known alias pattern
            if key.startswith("model.decoder."):
                short_key = key.replace("model.decoder.", "", 1)
                # If the short version exists in loaded state, this is just an alias - OK
                if short_key not in state_dict["module"]:
                    unexpected_missing.append(key)
            else:
                unexpected_missing.append(key)

        if unexpected_missing:
            logger.warning(
                "[EMA] Unexpected missing keys when loading EMA state: %s",
                unexpected_missing[:10],  # Show first 10
            )

        if unexpected_keys:
            logger.warning(
                "[EMA] Unexpected keys in loaded EMA state: %s",
                unexpected_keys[:10],  # Show first 10
            )

        self.updates = state_dict.get("updates", 0)
        self.decay = state_dict.get("decay", self.decay)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        logger.info(
            "[EMA] Loaded state: updates=%d, decay=%.4f, warmup_steps=%d",
            self.updates,
            self.decay,
            self.warmup_steps,
        )


class EMACallback(TrainerCallback):
    """Trainer callback to maintain EMA weights during training.

    This callback:
    - Creates EMA model copy at training start
    - Updates EMA after each optimizer step
    - Swaps to EMA weights during evaluation
    - Saves/loads EMA state with checkpoints

    Args:
        decay: EMA decay rate (default: 0.9999 from D-FINE)
        warmup_steps: Warmup period in steps (default: 1000 from D-FINE)
        use_ema_for_eval: Whether to use EMA weights for evaluation (default: True)

    Example:
        ```python
        from transformers import Trainer
        from doc_obj_detect.training.callbacks import EMACallback

        trainer = Trainer(
            model=model,
            args=training_args,
            callbacks=[EMACallback(decay=0.9999, warmup_steps=1000)],
        )
        trainer.train()
        ```
    """

    def __init__(
        self,
        decay: float = 0.9999,
        warmup_steps: int = 1000,
        use_ema_for_eval: bool = True,
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.use_ema_for_eval = use_ema_for_eval
        self.ema: ModelEMA | None = None
        self._original_model: nn.Module | None = None
        self._is_swapped = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        """Initialize EMA model at training start."""
        if model is None:
            logger.warning("[EMA] No model provided, skipping EMA initialization")
            return control

        # Create EMA model
        device = next(model.parameters()).device
        self.ema = ModelEMA(
            model=model,
            decay=self.decay,
            warmup_steps=self.warmup_steps,
            device=device,
        )

        # Restore EMA state if resuming from checkpoint
        from pathlib import Path

        ema_state_loaded = False

        # Check if we're loading from a specific checkpoint (from --load or --resume flags)
        # The model would have already been loaded from this checkpoint
        # Look for the checkpoint path in the training state
        checkpoint_to_load = None

        # Try to get checkpoint from TrainingArguments
        if hasattr(args, '_checkpoint_for_ema'):
            checkpoint_to_load = args._checkpoint_for_ema
        elif hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
            checkpoint_to_load = args.resume_from_checkpoint

        if checkpoint_to_load:
            # Load EMA from the specified checkpoint
            ema_path = Path(checkpoint_to_load) / "ema_state.pt"
            if ema_path.exists():
                try:
                    import torch

                    ema_state_dict = torch.load(ema_path, map_location=device)
                    self.ema.load_state_dict(ema_state_dict)
                    logger.info(
                        "[EMA] Restored EMA state from specified checkpoint: %s (updates=%d)",
                        ema_path,
                        self.ema.updates,
                    )
                    ema_state_loaded = True
                except Exception as e:
                    logger.warning(
                        "[EMA] Failed to load EMA state from %s: %s", ema_path, e
                    )

        # Fallback: if no specific checkpoint, try to find the most recent one in output_dir
        if not ema_state_loaded and args.output_dir:
            output_path = Path(args.output_dir)

            # Try to find the most recent checkpoint with EMA state
            checkpoint_dirs = sorted(output_path.glob("checkpoint-*"))
            if checkpoint_dirs:
                # Start from the most recent checkpoint
                for checkpoint_dir in reversed(checkpoint_dirs):
                    ema_path = checkpoint_dir / "ema_state.pt"
                    if ema_path.exists():
                        try:
                            import torch

                            ema_state_dict = torch.load(ema_path, map_location=device)
                            self.ema.load_state_dict(ema_state_dict)
                            logger.info(
                                "[EMA] Restored EMA state from latest checkpoint: %s (updates=%d)",
                                ema_path,
                                self.ema.updates,
                            )
                            ema_state_loaded = True
                            break
                        except Exception as e:
                            logger.warning(
                                "[EMA] Failed to load EMA state from %s: %s", ema_path, e
                            )

        if not ema_state_loaded:
            logger.info(
                "[EMA] Callback initialized from scratch (decay=%.4f, warmup=%d, use_for_eval=%s)",
                self.decay,
                self.warmup_steps,
                self.use_ema_for_eval,
            )
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        """Update EMA after each optimizer step."""
        if self.ema is not None and model is not None:
            self.ema.update(model)
        return control

    def on_evaluate(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        """Swap to EMA weights before evaluation."""
        if not self.use_ema_for_eval or self.ema is None or model is None:
            return control

        # Save original model and swap to EMA
        self._original_model = model
        self._swap_to_ema(kwargs.get("trainer"))
        self._is_swapped = True

        logger.debug("[EMA] Swapped to EMA weights for evaluation")
        return control

    def on_prediction_step(self, args, state, control, **kwargs):  # type: ignore[override]
        """Called after evaluation is complete - swap back to training weights."""
        if self._is_swapped and self._original_model is not None:
            self._swap_back_from_ema(kwargs.get("trainer"))
            self._is_swapped = False
            logger.debug("[EMA] Restored training weights after evaluation")
        return control

    def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
        """Save EMA state with checkpoint."""
        if self.ema is None:
            return control

        # Save EMA state alongside model checkpoint
        from pathlib import Path

        checkpoint_folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_folder.exists():
            ema_path = checkpoint_folder / "ema_state.pt"
            torch.save(self.ema.state_dict(), ema_path)
            logger.info("[EMA] Saved EMA state to %s", ema_path)

        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        """Save final EMA state."""
        if self.ema is None:
            return control

        from pathlib import Path

        ema_path = Path(args.output_dir) / "ema_state.pt"
        torch.save(self.ema.state_dict(), ema_path)
        logger.info("[EMA] Saved final EMA state to %s", ema_path)

        return control

    def _swap_to_ema(self, trainer) -> None:
        """Replace trainer's model with EMA model."""
        if trainer is not None and self.ema is not None:
            trainer.model = self.ema.module

    def _swap_back_from_ema(self, trainer) -> None:
        """Restore original training model."""
        if trainer is not None and self._original_model is not None:
            trainer.model = self._original_model
            self._original_model = None
