#!/usr/bin/env python3
"""Evaluate a checkpoint on the full PubLayNet validation set using COCO mAP metrics."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, DFineForObjectDetection
from transformers.trainer_utils import EvalPrediction

from doc_obj_detect.config import load_train_config
from doc_obj_detect.data import collate_fn, prepare_dataset_for_training
from doc_obj_detect.metrics import compute_map


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_eval_samples: int | None = None,
) -> dict:
    """Evaluate a checkpoint on the full validation set.

    Args:
        checkpoint_path: Path to checkpoint directory
        config_path: Path to training config YAML
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers
        max_eval_samples: Maximum number of samples to evaluate (None = full set)

    Returns:
        Dictionary of evaluation metrics
    """
    # Load config
    config = load_train_config(config_path)

    print("=" * 80)
    print(f"Evaluating Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model and processor
    print("\nLoading model from checkpoint...")
    model = DFineForObjectDetection.from_pretrained(checkpoint_path)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    aug_model = config.augmentation.model_dump() if config.augmentation else None
    if aug_model and aug_model.get("multi_scale_sizes"):
        eval_short_side = max(aug_model["multi_scale_sizes"])
    else:
        eval_short_side = config.data.image_size
    eval_long_side = aug_model.get("max_long_side") if aug_model else None
    eval_size = {"shortest_edge": eval_short_side}
    if eval_long_side:
        eval_size["longest_edge"] = eval_long_side
    image_processor.size = eval_size
    image_processor.do_resize = True
    model.to(device)
    model.eval()

    # Prepare validation dataset (full set, no max_samples limit)
    print("\nPreparing validation dataset...")
    data_config = config.data

    detector_stride = max(config.dfine.feat_strides)

    val_dataset, class_labels = prepare_dataset_for_training(
        dataset_name=data_config.dataset,
        split=data_config.val_split,
        image_processor=image_processor,
        augmentation_config=None,  # No augmentation for eval
        cache_dir=data_config.cache_dir,
        max_samples=max_eval_samples,  # None = full validation set
        pad_stride=detector_stride,
    )

    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Classes: {class_labels}")
    print(f"Batch size: {batch_size}")

    # Run evaluation manually to avoid Trainer's prediction accumulation OOM
    print("\n" + "=" * 80)
    print("Running Evaluation...")
    print("=" * 80 + "\n")

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Accumulate predictions and targets in chunks
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )

            # Accumulate loss
            total_loss += outputs.loss.item()
            num_batches += 1

            # Store predictions and targets (on CPU to save GPU memory)
            logits = outputs.logits.cpu()
            pred_boxes = outputs.pred_boxes.cpu()
            all_predictions.append((logits, pred_boxes))

            # Move targets back to CPU for metrics computation
            cpu_labels = [
                {
                    "class_labels": t["class_labels"].cpu().numpy(),
                    "boxes": t["boxes"].cpu().numpy(),
                    "orig_size": t["orig_size"].cpu().numpy(),
                }
                for t in labels
            ]
            all_targets.append(cpu_labels)

            # Free GPU memory
            del outputs, pixel_values, pixel_mask, labels
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches

    # Compute metrics using accumulated predictions
    print("\nComputing mAP metrics...")
    eval_pred = EvalPrediction(
        predictions=all_predictions,
        label_ids=all_targets,
    )

    metrics = compute_map(
        eval_pred=eval_pred,
        image_processor=image_processor,
        id2label=class_labels,
        threshold=0.0,  # Post-process all detections, NMS in post_process
        max_eval_images=max_eval_samples,  # Limit for metric computation
    )

    # Add loss to metrics
    metrics["eval_loss"] = avg_loss

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Validation samples: {len(val_dataset):,}")
    print("\nCOCO mAP Metrics:")
    print("-" * 80)

    # Pretty print metrics
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

    for key, name in metric_names.items():
        if key in metrics:
            print(f"{name:.<40} {metrics[key]:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint on full PubLayNet validation set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., outputs/pretrain_publaynet_dfine/checkpoint-13500)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_publaynet.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: None = full set)",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Run evaluation
    evaluate_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config_path=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_samples=args.max_samples,
    )

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
