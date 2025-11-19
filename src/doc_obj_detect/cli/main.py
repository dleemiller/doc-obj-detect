"""Central CLI dispatch for training, evaluation, and utilities."""

from __future__ import annotations

import argparse
import logging

from doc_obj_detect.data.datasets import DatasetLoader
from doc_obj_detect.tools import (
    collect_bbox_samples,
    compute_stride_ranges,
    load_bbox_data,
    plot_bbox_histograms,
    save_bbox_data,
)
from doc_obj_detect.training import DistillRunner, EvaluatorRunner, TrainerRunner
from doc_obj_detect.visualize import visualize_augmentations

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="doc-obj-detect", description="Unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_distill_parser(subparsers)
    _add_visualize_parser(subparsers)
    _add_dataset_info_parser(subparsers)
    _add_bbox_hist_parser(subparsers)

    args = parser.parse_args(argv)
    args.handler(args)


# ---------------------------------------------------------------------------
# Subcommand builders
# ---------------------------------------------------------------------------


def _add_train_parser(subparsers):
    parser = subparsers.add_parser("train", help="Run D-FINE training")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.set_defaults(handler=_handle_train)


def _add_evaluate_parser(subparsers):
    parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.set_defaults(handler=_handle_evaluate)


def _add_distill_parser(subparsers):
    parser = subparsers.add_parser("distill", help="Run knowledge distillation")
    parser.add_argument("--config", required=True, help="Path to distillation YAML config")
    parser.set_defaults(handler=_handle_distill)


def _add_visualize_parser(subparsers):
    parser = subparsers.add_parser("visualize", help="Preview augmentations")
    parser.add_argument("--dataset", choices=["publaynet", "doclaynet"], required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="outputs/augmentation_samples")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.set_defaults(handler=_handle_visualize)


def _add_dataset_info_parser(subparsers):
    parser = subparsers.add_parser("dataset-info", help="Quick dataset summary")
    parser.add_argument("--dataset", choices=["publaynet", "doclaynet"], required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.set_defaults(handler=_handle_dataset_info)


def _add_bbox_hist_parser(subparsers):
    parser = subparsers.add_parser("bbox-hist", help="Analyze bbox widths/heights")
    parser.add_argument(
        "--config",
        required=False,
        help="Path to training YAML config (required unless --data-in provided)",
    )
    parser.add_argument("--split", default="val", help="Dataset split to analyze")
    parser.add_argument("--output", default="outputs/bbox_hist.png", help="Output figure path")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins")
    parser.add_argument(
        "--short-side",
        type=int,
        default=640,
        help="Override short side resize (defaults to config's evaluation short side)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Limit samples (validation/test splits only)",
    )
    parser.add_argument(
        "--data-out",
        type=str,
        default="bbox_stats.npz",
        help="Optional path to save collected bbox statistics (.npz)",
    )
    parser.add_argument(
        "--data-in",
        type=str,
        default=None,
        help="Load precomputed bbox statistics (.npz) instead of scanning the dataset",
    )
    parser.set_defaults(handler=_handle_bbox_hist)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_train(args):
    runner = TrainerRunner.from_config(args.config)
    runner.run()


def _handle_evaluate(args):
    runner = EvaluatorRunner.from_config(args.config)
    runner.run(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_samples=args.max_samples,
    )


def _handle_distill(args):
    runner = DistillRunner.from_config(args.config)
    runner.run()


def _handle_visualize(args):
    visualize_augmentations(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )


def _handle_dataset_info(args):
    dataset = args.dataset
    cache_dir = args.cache_dir
    logger.info("Loading %s dataset...", dataset)
    if dataset == "publaynet":
        train_ds, labels = DatasetLoader.load_publaynet("train", cache_dir)
        val_ds, _ = DatasetLoader.load_publaynet("val", cache_dir)
    else:
        train_ds, labels = DatasetLoader.load_doclaynet("train", cache_dir)
        val_ds, _ = DatasetLoader.load_doclaynet("val", cache_dir)

    logger.info("Dataset loaded successfully!")
    logger.info("Train samples: %s", len(train_ds))
    logger.info("Val samples: %s", len(val_ds))
    logger.info("Classes: %s", labels)


def _handle_bbox_hist(args):
    if args.data_in:
        widths, heights, strides = load_bbox_data(args.data_in)
        logger.info("Loaded bbox data from %s", args.data_in)
    else:
        if not args.config:
            raise ValueError("--config is required when --data-in is not provided.")
        widths, heights, strides = collect_bbox_samples(
            config_path=args.config,
            split=args.split,
            max_samples=args.max_samples,
            target_short_side=args.short_side,
        )
        logger.info(
            "Collected %s boxes from %s split using short side %s",
            len(widths),
            args.split,
            args.short_side or "config default",
        )
        if args.data_out:
            save_bbox_data(args.data_out, widths, heights, strides)
            logger.info("Saved bbox data to %s", args.data_out)

    stride_ranges = compute_stride_ranges(strides)
    for stride, rng in stride_ranges.items():
        logger.info("Stride %s covers approx %.1fpx - %.1fpx", stride, rng[0], rng[1])

    plot_bbox_histograms(
        widths=widths,
        heights=heights,
        stride_ranges=stride_ranges,
        strides=strides,
        bins=args.bins,
        output_path=args.output,
    )
    logger.info("Saved bbox histograms to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
