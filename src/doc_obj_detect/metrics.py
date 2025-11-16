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
    max_eval_images: int | None = 2000,
) -> dict[str, float]:
    """
    Compute COCO-style mAP/mAR with torchmetrics, in a memory-aware way.

    - Uses HF's object detection recipe.
    - Optionally caps number of images used for metrics to avoid OOM.
    """
    predictions, targets = eval_pred.predictions, eval_pred.label_ids

    # Safety: cap number of images considered by metrics
    predictions, targets = _limit_eval_images(
        list(predictions),
        list(targets),
        max_images=max_eval_images,
    )

    image_sizes: list[torch.Tensor] = []
    processed_targets: list[dict[str, torch.Tensor]] = []
    processed_predictions: list[dict[str, torch.Tensor]] = []

    # --- targets: YOLO -> Pascal VOC on CPU ---
    for batch in targets:
        # batch is list[dict], each dict has "orig_size", "boxes", "class_labels"
        batch_image_sizes = torch.tensor(
            np.array([t["orig_size"] for t in batch]),
            dtype=torch.int64,
        )
        image_sizes.append(batch_image_sizes)

        for t in batch:
            boxes = torch.tensor(t["boxes"], dtype=torch.float32)
            boxes = convert_bbox_yolo_to_pascal(boxes, t["orig_size"])
            labels = torch.tensor(t["class_labels"], dtype=torch.int64)
            processed_targets.append({"boxes": boxes, "labels": labels})

    # --- predictions: model YOLO boxes -> Pascal VOC via image_processor ---
    for batch_pred, target_sizes in zip(predictions, image_sizes, strict=False):
        # HF Deformable DETR: predictions per batch are (loss, logits, boxes) or (logits, boxes)
        if len(batch_pred) == 3:
            _, batch_logits, batch_boxes = batch_pred
        else:
            batch_logits, batch_boxes = batch_pred

        output = ModelOutput(
            logits=torch.tensor(batch_logits),  # keep on CPU
            pred_boxes=torch.tensor(batch_boxes),
        )

        post = image_processor.post_process_object_detection(
            output,
            threshold=threshold,
            target_sizes=target_sizes,
        )
        processed_predictions.extend(post)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        class_metrics=True,
    )
    metric.update(processed_predictions, processed_targets)
    metrics = metric.compute()

    # per-class expansion
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")

    for cls_id, cls_map, cls_mar in zip(classes, map_per_class, mar_100_per_class, strict=False):
        name = id2label[cls_id.item()] if id2label is not None else cls_id.item()
        metrics[f"map_{name}"] = cls_map
        metrics[f"mar_100_{name}"] = cls_mar

    # tensor -> float
    return {k: float(v.item()) for k, v in metrics.items()}
