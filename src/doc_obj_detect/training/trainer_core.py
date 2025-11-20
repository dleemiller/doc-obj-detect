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
    """

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
        # ConvNeXt-Large DINOv3 (198M params) vs HGNetV2 (4-40M params):
        # D-FINE-X (40M backbone): 0.01 multiplier
        # D-FINE-L (18M backbone): 0.05 multiplier
        # ConvNeXt-L is 5-10× larger + DINOv3 pretrained on 1.7B images
        # Using 0.01 (conservative) to preserve strong DINOv3 representations
        optimizer_grouped_parameters = [
            {"params": other_params, "lr": base_lr},
            {"params": backbone_params, "lr": base_lr * 0.01},  # 2.5e-6 for base_lr=2.5e-4
        ]

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

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
