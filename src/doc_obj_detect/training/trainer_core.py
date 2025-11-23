"""Custom HuggingFace Trainer extensions."""

from __future__ import annotations

import logging

import torch
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

logger = logging.getLogger(__name__)


class SplitLRTrainer(Trainer):
    """Trainer variant that uses split learning rates for backbone vs head.

    In addition to the optimizer change, evaluation is overridden so we can
    accumulate raw logits/boxes for document-detection metrics without flattening
    nested tensors the way the stock Trainer does.

    Attributes:
        backbone_lr_multiplier: Learning rate multiplier for backbone parameters.
            Defaults to 0.01 (1% of head LR) to preserve pretrained representations.
        backbone_max_grad_norm: Gradient clipping norm for backbone parameters.
            If None, uses args.max_grad_norm.
        head_max_grad_norm: Gradient clipping norm for head parameters.
            If None, uses args.max_grad_norm.
    """

    def __init__(
        self,
        *args,
        backbone_lr_multiplier: float = 0.01,
        backbone_max_grad_norm: float | None = None,
        head_max_grad_norm: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.backbone_lr_multiplier = backbone_lr_multiplier

        # Use separate norms if provided, otherwise fall back to global max_grad_norm
        self.backbone_max_grad_norm = backbone_max_grad_norm or self.args.max_grad_norm
        self.head_max_grad_norm = head_max_grad_norm or self.args.max_grad_norm
        self.use_separate_grad_norms = (
            backbone_max_grad_norm is not None or head_max_grad_norm is not None
        )

        logger.info(
            "SplitLRTrainer initialized with backbone_lr_multiplier=%.4f",
            self.backbone_lr_multiplier,
        )
        if self.use_separate_grad_norms:
            logger.info(
                "Using separate gradient clipping: backbone=%.3f, head=%.3f",
                self.backbone_max_grad_norm,
                self.head_max_grad_norm,
            )
        else:
            logger.info(
                "Using global gradient clipping: max_grad_norm=%.3f", self.args.max_grad_norm
            )

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is not None:
            return self.optimizer

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "model.backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        base_lr = self.args.learning_rate
        backbone_lr = base_lr * self.backbone_lr_multiplier

        # Count actual parameters (scalars), not just tensors
        backbone_param_count = sum(p.numel() for p in backbone_params)
        head_param_count = sum(p.numel() for p in other_params)

        logger.info("Creating optimizer with split learning rates:")
        logger.info("  Head LR: %.2e", base_lr)
        logger.info(
            "  Backbone LR: %.2e (%.2f%% of head LR)",
            backbone_lr,
            self.backbone_lr_multiplier * 100,
        )
        logger.info("  Backbone params: %s", f"{backbone_param_count:,}")
        logger.info("  Head params: %s", f"{head_param_count:,}")

        optimizer_grouped_parameters = [
            {"params": other_params, "lr": base_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        """Override training step to add custom gradient clipping."""
        # Call parent training_step which handles forward, backward, etc.
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Apply custom gradient clipping after backward pass
        if self.use_separate_grad_norms:
            self._clip_gradients_custom(model)

        return loss

    def _clip_gradients_custom(self, model):
        """Custom gradient clipping that separates backbone and head gradients.

        This prevents the large backbone gradients from dominating the gradient budget
        and starving the head of updates when global gradient clipping is applied.
        """
        # Unwrap model if using DDP/FSDP
        if hasattr(model, "module"):
            model = model.module

        # Separate parameters into backbone vs head
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if "model.backbone" in name or "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # Clip each group independently
        # clip_grad_norm_ returns the TOTAL NORM BEFORE CLIPPING
        backbone_norm_before = 0.0
        head_norm_before = 0.0

        if backbone_params:
            backbone_norm_before = torch.nn.utils.clip_grad_norm_(
                backbone_params, max_norm=self.backbone_max_grad_norm
            ).item()

        if head_params:
            head_norm_before = torch.nn.utils.clip_grad_norm_(
                head_params, max_norm=self.head_max_grad_norm
            ).item()

        # Log gradient norms periodically (every 100 steps)
        if self.state.global_step % 100 == 0:
            logger.info(
                "Gradient norms BEFORE clipping: backbone=%.3f, head=%.3f",
                backbone_norm_before,
                head_norm_before,
            )
            logger.info(
                "Clipping to: backbone_max=%.3f, head_max=%.3f",
                self.backbone_max_grad_norm,
                self.head_max_grad_norm,
            )

            # Log clipping actions
            if backbone_norm_before > self.backbone_max_grad_norm:
                logger.info(
                    "  → Backbone gradients CLIPPED (%.3f → %.3f, scale=%.3f)",
                    backbone_norm_before,
                    self.backbone_max_grad_norm,
                    self.backbone_max_grad_norm / backbone_norm_before,
                )
            elif backbone_norm_before > 0:
                logger.info(
                    "  → Backbone gradients unchanged (%.3f < %.3f)",
                    backbone_norm_before,
                    self.backbone_max_grad_norm,
                )

            if head_norm_before > self.head_max_grad_norm:
                logger.info(
                    "  → Head gradients CLIPPED (%.3f → %.3f, scale=%.3f)",
                    head_norm_before,
                    self.head_max_grad_norm,
                    self.head_max_grad_norm / head_norm_before,
                )
            elif head_norm_before > 0:
                logger.info(
                    "  → Head gradients unchanged (%.3f < %.3f)",
                    head_norm_before,
                    self.head_max_grad_norm,
                )

    def get_eval_dataloader(self, eval_dataset=None):  # type: ignore[override]
        original_num_workers = self.args.dataloader_num_workers
        self.args.dataloader_num_workers = 0
        eval_dataloader = super().get_eval_dataloader(eval_dataset)
        self.args.dataloader_num_workers = original_num_workers
        return eval_dataloader

    def evaluation_loop(  # type: ignore[override]
        self,
        dataloader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        all_preds = []
        all_labels = []
        all_losses = []

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs.loss.mean().item() if outputs.loss is not None else None
            if loss is not None:
                all_losses.append(loss)

            batch_pred = (
                outputs.logits.cpu().numpy(),
                outputs.pred_boxes.cpu().numpy(),
            )
            all_preds.append(batch_pred)

            batch_labels = []
            for i in range(len(inputs["pixel_values"])):
                label_dict = {
                    "class_labels": inputs["labels"][i]["class_labels"].cpu().numpy(),
                    "boxes": inputs["labels"][i]["boxes"].cpu().numpy(),
                    "orig_size": inputs["labels"][i]["orig_size"].cpu().numpy(),
                }
                batch_labels.append(label_dict)
            all_labels.append(batch_labels)

        metrics = {}
        metric_prefix = f"{metric_key_prefix}_" if metric_key_prefix else ""
        if self.compute_metrics is not None:
            eval_pred = EvalPrediction(predictions=all_preds, label_ids=all_labels)
            metrics = self.compute_metrics(eval_pred)
            if metric_prefix:
                metrics = {
                    key if key.startswith(metric_prefix) else f"{metric_prefix}{key}": value
                    for key, value in metrics.items()
                }

        if all_losses:
            metrics[f"{metric_key_prefix}_loss"] = sum(all_losses) / len(all_losses)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=len(dataloader.dataset),
        )

    def log(self, logs, start_time=None):  # type: ignore[override]
        """Log extra diagnostics such as VRAM usage."""

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            try:
                mem_allocated = torch.cuda.max_memory_allocated(device)
                mem_reserved = torch.cuda.memory_reserved(device)
                mem_current = torch.cuda.memory_allocated(device)
            except RuntimeError:
                mem_allocated = mem_reserved = mem_current = 0

            if mem_allocated > 0:
                metric_prefix = "eval" if any(key.startswith("eval") for key in logs) else "train"
                logs.setdefault(f"{metric_prefix}_vram_peak_mb", mem_allocated / (1024**2))
                logs.setdefault(f"{metric_prefix}_vram_reserved_mb", mem_reserved / (1024**2))
                logs.setdefault(f"{metric_prefix}_vram_current_mb", mem_current / (1024**2))
            torch.cuda.reset_peak_memory_stats(device)

        return super().log(logs, start_time)
