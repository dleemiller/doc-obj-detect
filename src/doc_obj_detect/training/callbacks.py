"""Custom Trainer callbacks used by the training runner."""

import logging

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
