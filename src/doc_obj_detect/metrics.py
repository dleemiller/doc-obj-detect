# doc_obj_detect/metrics.py

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format
from transformers.trainer_utils import EvalPrediction


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """
    YOLO -> Pascal VOC (x_min, y_min, x_max, y_max) in absolute pixel coords.
    boxes: [num_boxes, 4] in [0, 1] (cx, cy, w, h)
    image_size: (H, W)
    """
    # center -> corners
    boxes = center_to_corners_format(boxes)
    h, w = image_size
    boxes = boxes * torch.tensor([[w, h, w, h]], dtype=boxes.dtype)
    return boxes


def _limit_eval_images(
    predictions: list[Any],
    targets: list[Any],
    max_images: int,
) -> tuple[list[Any], list[Any]]:
    """
    Reduce the number of images used for metrics to avoid OOM.
    predictions: list over batches; each batch is (loss?, logits, pred_boxes)
    targets: list over batches; each batch is a list[dict] (one per image)
    """
    if max_images is None:
        return predictions, targets

    new_preds: list[Any] = []
    new_tgts: list[Any] = []
    count = 0

    for batch_pred, batch_tgt in zip(predictions, targets, strict=False):
        if count >= max_images:
            break

        batch_size = len(batch_tgt)
        remaining = max_images - count

        if batch_size <= remaining:
            # keep whole batch
            new_preds.append(batch_pred)
            new_tgts.append(batch_tgt)
            count += batch_size
        else:
            # need to slice this batch
            # batch_pred is something like (loss, logits, boxes)
            # or just (logits, boxes) depending on HF version
            if len(batch_pred) == 3:
                loss, logits, boxes = batch_pred
                loss = loss[:remaining] if loss is not None else None
                logits = logits[:remaining]
                boxes = boxes[:remaining]
                sliced_pred = (loss, logits, boxes)
            else:
                logits, boxes = batch_pred
                sliced_pred = (logits[:remaining], boxes[:remaining])

            new_preds.append(sliced_pred)
            new_tgts.append(batch_tgt[:remaining])
            count += remaining

    return new_preds, new_tgts


@torch.no_grad()
def compute_map(
    eval_pred: EvalPrediction,
    image_processor,
    id2label: Mapping[int, str] | None = None,
    threshold: float = 0.0,
    max_eval_images: int | None = 200,  # cap for metric compute to avoid OOM
) -> dict[str, float]:
    """
    Compute COCO-style mAP/mAR with torchmetrics on CPU.

    - Caps the number of images via `max_eval_images` for speed/memory.
    - Uses HF's post_process_object_detection for prediction decoding.
    - Uses YOLO -> Pascal VOC conversion for targets.
    - Runs MeanAveragePrecision on CPU with class_metrics=False for speed.
    """
    predictions, targets = eval_pred.predictions, eval_pred.label_ids

    # Safety: cap number of images considered by metrics
    predictions, targets = _limit_eval_images(
        list(predictions),
        list(targets),
        max_images=max_eval_images,
    )

    # First, process all predictions and targets on CPU
    processed_targets_cpu: list[dict[str, torch.Tensor]] = []
    processed_predictions_cpu: list[dict[str, torch.Tensor]] = []

    for batch_pred, batch_tgt in zip(predictions, targets, strict=False):
        # Extract prediction tensors
        if len(batch_pred) == 3:
            _, batch_logits, batch_boxes = batch_pred
        else:
            batch_logits, batch_boxes = batch_pred

        # Convert to torch tensors once
        logits_tensor = (
            batch_logits
            if isinstance(batch_logits, torch.Tensor)
            else torch.from_numpy(batch_logits)
        )
        boxes_tensor = (
            batch_boxes if isinstance(batch_boxes, torch.Tensor) else torch.from_numpy(batch_boxes)
        )

        # Batch image sizes for post-processing
        target_sizes = torch.tensor(
            np.array([t["orig_size"] for t in batch_tgt]),
            dtype=torch.int64,
        )

        # Post-process predictions (CPU)
        output = ModelOutput(logits=logits_tensor, pred_boxes=boxes_tensor)
        post = image_processor.post_process_object_detection(
            output,
            threshold=threshold,
            target_sizes=target_sizes,
        )
        processed_predictions_cpu.extend(post)

        # Process targets (YOLO → Pascal VOC) on CPU
        for t in batch_tgt:
            boxes = torch.as_tensor(t["boxes"], dtype=torch.float32)
            boxes = convert_bbox_yolo_to_pascal(boxes, t["orig_size"])
            labels = torch.as_tensor(t["class_labels"], dtype=torch.int64)
            processed_targets_cpu.append({"boxes": boxes, "labels": labels})

    # Metric on CPU, no per-class metrics for speed
    metric = MeanAveragePrecision(
        box_format="xyxy",
        class_metrics=False,
    )
    # Allow >100 detections/image without noisy warnings (we rely on HF post-processing thresholds)
    metric.warn_on_many_detections = False

    # Single update with all images (capped by max_eval_images)
    metric.update(processed_predictions_cpu, processed_targets_cpu)

    # Compute final metrics
    metrics = metric.compute()

    # Drop non-scalar fields we don't need (present even when class_metrics=False)
    metrics.pop("classes", None)
    metrics.pop("map_per_class", None)
    metrics.pop("mar_100_per_class", None)

    # tensor -> float (all tensors should now be 0-dim scalars)
    return {k: float(v.item()) for k, v in metrics.items()}
