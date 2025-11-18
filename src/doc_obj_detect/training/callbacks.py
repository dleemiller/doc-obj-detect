"""Custom Trainer callbacks used by the training runner."""

import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class UnfreezeBackboneCallback(TrainerCallback):
    """Unfreeze the backbone after a warmup period.

    HuggingFace's Trainer API does not provide a built-in mechanism to partially
    unfreeze parameters after a number of steps. This callback watches the
    training state and flips ``requires_grad`` for the backbone once the desired
    step is reached.
    """

    def __init__(self, unfreeze_at_step: int) -> None:
        self.unfreeze_at_step = unfreeze_at_step
        self._done = False

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if self._done:
            return control

        if state.global_step >= self.unfreeze_at_step:
            model = kwargs.get("model")
            if model is None:
                return control

            backbone = getattr(model, "model", None)
            backbone = getattr(backbone, "backbone", None)
            if backbone is None:
                return control

            for param in backbone.parameters():
                param.requires_grad = True

            logger.info("[Callback] Unfroze backbone at step %s", state.global_step)
            self._done = True
        return control
