"""Evaluation and metrics for object detection models."""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

from doc_obj_detect.config import load_train_config
from doc_obj_detect.data import collate_fn, prepare_dataset_for_training


def compute_map(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mean Average Precision (mAP) for object detection.

    This is a simplified mAP implementation. For production use, consider
    using pycocotools.coco.COCOeval for comprehensive evaluation.

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        targets: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for considering a detection as correct

    Returns:
        Dict with mAP metrics
    """
    from torchvision.ops import box_iou

    total_tp = 0
    total_fp = 0
    total_gt = sum(len(t["boxes"]) for t in targets)

    for pred, target in zip(predictions, targets, strict=False):
        if len(pred["boxes"]) == 0:
            continue

        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        target_boxes = target["boxes"]
        target_labels = target["labels"]

        # Compute IoU between all predicted and ground truth boxes
        ious = box_iou(pred_boxes, target_boxes)

        matched_gt = set()
        for i, pred_label in enumerate(pred_labels):
            max_iou, max_idx = ious[i].max(dim=0)

            if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
                # Check if labels match
                if pred_label == target_labels[max_idx]:
                    total_tp += 1
                    matched_gt.add(max_idx.item())
                else:
                    total_fp += 1
            else:
                total_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP": precision,  # Simplified - actual mAP requires AP per class
    }


def evaluate_model(
    model: DeformableDetrForObjectDetection,
    dataloader: DataLoader,
    image_processor: AutoImageProcessor,
    device: torch.device,
    confidence_threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate object detection model.

    Args:
        model: Trained detection model
        dataloader: Evaluation data loader
        image_processor: Image processor for post-processing
        device: Device to run evaluation on
        confidence_threshold: Minimum confidence for predictions

    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            # Forward pass
            outputs = model(pixel_values=pixel_values)

            # Post-process predictions
            target_sizes = torch.stack(
                [torch.tensor([img.shape[1], img.shape[2]], device=device) for img in pixel_values]
            )

            results = image_processor.post_process_object_detection(
                outputs,
                threshold=confidence_threshold,
                target_sizes=target_sizes,
            )

            # Convert to evaluation format
            for result, label in zip(results, labels, strict=False):
                pred = {
                    "boxes": result["boxes"].cpu(),
                    "scores": result["scores"].cpu(),
                    "labels": result["labels"].cpu(),
                }
                target = {
                    "boxes": label["boxes"].cpu(),
                    "labels": label["class_labels"].cpu(),
                }
                all_predictions.append(pred)
                all_targets.append(target)

    # Compute metrics
    metrics = compute_map(all_predictions, all_targets)

    return metrics


def evaluate(config_path: str, checkpoint_path: str) -> None:
    """Evaluate a trained model.

    Args:
        config_path: Path to training configuration YAML file
        checkpoint_path: Path to model checkpoint
    """
    # Load and validate configuration
    config = load_train_config(config_path)

    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load model and processor
    print(f"\nLoading model from: {checkpoint_path}")
    model = DeformableDetrForObjectDetection.from_pretrained(checkpoint_path)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    model.to(device)

    # Prepare dataset
    print("\nPreparing evaluation dataset...")
    data_config = config.data

    eval_dataset, class_labels = prepare_dataset_for_training(
        dataset_name=data_config.dataset,
        split=data_config.test_split or data_config.val_split,
        image_processor=image_processor,
        augmentation_config=None,
        cache_dir=data_config.cache_dir,
    )

    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Classes: {class_labels}")

    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=data_config.batch_size,
        collate_fn=collate_fn,
        num_workers=data_config.num_workers,
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(
        model=model,
        dataloader=eval_dataloader,
        image_processor=image_processor,
        device=device,
        confidence_threshold=0.5,
    )

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate document object detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
