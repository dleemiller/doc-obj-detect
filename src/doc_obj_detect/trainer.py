import torch
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput


class SplitLRTrainer(Trainer):
    """
    Trainer with two learning rates:
      - base lr for everything except backbone
      - scaled-down lr for model.model.backbone parameters

    Also overrides eval dataloader to use num_workers=0 to avoid
    "too many open files" error when train and eval run concurrently.

    Handles prediction collection for object detection metrics.
    """

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Split parameters into backbone vs non-backbone
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "model.backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        # Get the usual optimizer class/kwargs from TrainingArguments
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

        # Two param groups: head uses self.args.learning_rate,
        # backbone uses a smaller lr (e.g., * 0.1)
        lr = self.args.learning_rate
        optimizer_grouped_parameters = [
            {
                "params": other_params,
                "lr": lr,
            },
            {
                "params": backbone_params,
                "lr": lr * 0.1,
            },
        ]

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Override to use num_workers=0 for eval to avoid file handle issues.
        When train and eval dataloaders run concurrently (during eval steps),
        having workers on both can exhaust file descriptors.
        """
        # Temporarily override num_workers for eval
        original_num_workers = self.args.dataloader_num_workers
        self.args.dataloader_num_workers = 0

        # Call parent implementation
        eval_dataloader = super().get_eval_dataloader(eval_dataset)

        # Restore original value
        self.args.dataloader_num_workers = original_num_workers

        return eval_dataloader

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """
        Override evaluation loop to properly collect predictions for object detection.
        The default Trainer tries to concatenate predictions which doesn't work for
        variable-length object detection outputs.
        """
        # Store predictions and labels as lists (not concatenated)
        all_preds = []
        all_labels = []
        all_losses = []

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        for _step, inputs in enumerate(dataloader):
            # Move to device
            inputs = self._prepare_inputs(inputs)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            # Extract loss
            loss = outputs.loss.mean().item() if outputs.loss is not None else None
            if loss is not None:
                all_losses.append(loss)

            # Store predictions: (logits, pred_boxes) per batch
            # outputs.logits: [batch, num_queries, num_classes]
            # outputs.pred_boxes: [batch, num_queries, 4]
            batch_pred = (
                outputs.logits.cpu().numpy(),
                outputs.pred_boxes.cpu().numpy(),
            )
            all_preds.append(batch_pred)

            # Store labels: list of dicts per batch
            # Each dict has keys: "class_labels", "boxes", "orig_size"
            batch_labels = []
            for i in range(len(inputs["pixel_values"])):
                label_dict = {
                    "class_labels": inputs["labels"][i]["class_labels"].cpu().numpy(),
                    "boxes": inputs["labels"][i]["boxes"].cpu().numpy(),
                    "orig_size": inputs["labels"][i]["orig_size"].cpu().numpy(),
                }
                batch_labels.append(label_dict)
            all_labels.append(batch_labels)

        # Compute metrics if compute_metrics is defined
        metrics = {}
        if self.compute_metrics is not None:
            # Create EvalPrediction with list format (not concatenated)
            from transformers.trainer_utils import EvalPrediction

            eval_pred = EvalPrediction(predictions=all_preds, label_ids=all_labels)
            metrics = self.compute_metrics(eval_pred)

        # Add average loss
        if all_losses:
            metrics[f"{metric_key_prefix}_loss"] = sum(all_losses) / len(all_losses)

        # Prefix all metrics
        if metric_key_prefix != "eval":
            metrics = {
                f"{metric_key_prefix}_{k}" if not k.startswith(metric_key_prefix) else k: v
                for k, v in metrics.items()
            }

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=len(dataloader.dataset),
        )
