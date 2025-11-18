#!/usr/bin/env python3
"""Evaluate a checkpoint on the validation set using COCO mAP metrics."""

import argparse
from pathlib import Path

from doc_obj_detect.training import EvaluatorRunner


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_eval_samples: int | None = None,
) -> dict[str, float]:
    runner = EvaluatorRunner.from_config(config_path)
    return runner.run(
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_eval_samples=max_eval_samples,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on validation set")
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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
