"""Reusable runner for checkpoint evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, DFineForObjectDetection
from transformers.trainer_utils import EvalPrediction

from doc_obj_detect.config import TrainConfig, load_train_config
from doc_obj_detect.data import DatasetFactory, collate_fn
from doc_obj_detect.metrics import compute_map
from doc_obj_detect.utils import prepare_run_dirs, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    batch_size: int = 32
    num_workers: int = 4
    max_eval_samples: int | None = None


class EvaluatorRunner:
    def __init__(self, config: TrainConfig, config_path: str | Path | None = None) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self._aug_config = config.augmentation.model_dump() if config.augmentation else None
        self._detector_stride = max(config.dfine.feat_strides)

    @classmethod
    def from_config(cls, config_path: str | Path) -> EvaluatorRunner:
        cfg = load_train_config(config_path)
        return cls(cfg, config_path)

    def run(
        self,
        checkpoint_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        max_eval_samples: int | None = None,
    ) -> dict[str, float]:
        checkpoint_path = str(checkpoint_path)
        logger.info("=" * 80)
        logger.info("Evaluating Checkpoint: %s", checkpoint_path)
        logger.info("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)

        model, processor = self._load_checkpoint(checkpoint_path, device)
        run_paths = prepare_run_dirs(self.config.output)
        setup_logging(run_paths.run_name, run_paths.log_dir)

        dataset, class_labels = self._build_dataset(processor, max_eval_samples)
        dataloader = self._build_dataloader(dataset, batch_size, num_workers)

        metrics = self._evaluate_checkpoint(
            model=model,
            dataloader=dataloader,
            processor=processor,
            class_labels=class_labels,
            max_eval_samples=max_eval_samples,
            device=device,
        )

        self._pretty_print(metrics, checkpoint_path, len(dataset))
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_path: str, device: torch.device):
        logger.info("Loading model from checkpoint...")
        model = DFineForObjectDetection.from_pretrained(checkpoint_path)
        processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        processor.size = self._build_eval_size()
        processor.do_resize = True
        processor.do_pad = True
        model.to(device)
        model.eval()
        return model, processor

    def _build_eval_size(self) -> dict[str, int]:
        # Determine base size
        if self._aug_config and self._aug_config.get("multi_scale_sizes"):
            eval_short_side = max(self._aug_config["multi_scale_sizes"])
        else:
            eval_short_side = self.config.data.image_size

        # If training with square resize, eval should also use square
        if self._aug_config and self._aug_config.get("force_square_resize", False):
            return {"height": eval_short_side, "width": eval_short_side}

        # Otherwise use shortest_edge with optional max_long_side
        eval_size = {"shortest_edge": eval_short_side}
        if self._aug_config and self._aug_config.get("max_long_side"):
            eval_size["longest_edge"] = self._aug_config["max_long_side"]
        return eval_size

    def _build_dataset(self, processor, max_eval_samples: int | None):
        data_cfg = self.config.data
        logger.info("Preparing validation dataset...")
        factory = DatasetFactory(
            dataset_name=data_cfg.dataset,
            image_processor=processor,
            pad_stride=self._detector_stride,
            cache_dir=data_cfg.cache_dir,
            augmentation_config=None,
        )
        dataset, class_labels = factory.build(
            split=data_cfg.val_split,
            max_samples=max_eval_samples,
            apply_augmentation=False,
        )
        logger.info("Validation samples: %s", f"{len(dataset):,}")
        logger.info("Classes: %s", class_labels)
        return dataset, class_labels

    def _build_dataloader(self, dataset, batch_size: int, num_workers: int):
        logger.info("Batch size: %s", batch_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return dataloader

    def _evaluate_checkpoint(
        self,
        model,
        dataloader,
        processor,
        class_labels,
        max_eval_samples,
        device,
    ) -> dict[str, float]:
        logger.info("=" * 80)
        logger.info("Running Evaluation...")
        logger.info("=" * 80)

        all_predictions: list[Any] = []
        all_targets: list[Any] = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )

                loss = outputs.loss.item()
                total_loss += loss
                num_batches += 1

                logits = outputs.logits.cpu()
                pred_boxes = outputs.pred_boxes.cpu()
                all_predictions.append((logits, pred_boxes))

                cpu_labels = [
                    {
                        "class_labels": t["class_labels"].cpu().numpy(),
                        "boxes": t["boxes"].cpu().numpy(),
                        "orig_size": t["orig_size"].cpu().numpy(),
                    }
                    for t in labels
                ]
                all_targets.append(cpu_labels)

                del outputs, pixel_values, pixel_mask, labels
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, num_batches)

        eval_pred = EvalPrediction(
            predictions=all_predictions,
            label_ids=all_targets,
        )

        metrics = compute_map(
            eval_pred=eval_pred,
            image_processor=processor,
            id2label=class_labels,
            threshold=0.0,
            max_eval_images=max_eval_samples if max_eval_samples else 2000,
        )
        # Prefix all metrics with 'eval_' to match expected format
        eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
        eval_metrics["eval_loss"] = avg_loss
        return eval_metrics

    def _pretty_print(self, metrics: dict[str, float], checkpoint_path: str, num_samples: int):
        metric_names = {
            "eval_map": "mAP (IoU=0.50:0.95)",
            "eval_map_50": "mAP @ IoU=0.50",
            "eval_map_75": "mAP @ IoU=0.75",
            "eval_map_small": "mAP (small objects)",
            "eval_map_medium": "mAP (medium objects)",
            "eval_map_large": "mAP (large objects)",
            "eval_mar_1": "mAR @ 1 det/img",
            "eval_mar_10": "mAR @ 10 det/img",
            "eval_mar_100": "mAR @ 100 det/img",
            "eval_mar_small": "mAR (small objects)",
            "eval_mar_medium": "mAR (medium objects)",
            "eval_mar_large": "mAR (large objects)",
            "eval_loss": "Loss",
        }

        logger.info("=" * 80)
        logger.info("Evaluation Results")
        logger.info("=" * 80)
        logger.info("Checkpoint: %s", checkpoint_path)
        logger.info("Validation samples: %s", f"{num_samples:,}")
        logger.info("COCO mAP Metrics:")
        logger.info("-" * 80)
        for key, name in metric_names.items():
            if key in metrics:
                logger.info("%s %0.4f", f"{name:.<40}", metrics[key])


__all__ = ["EvaluatorRunner"]
