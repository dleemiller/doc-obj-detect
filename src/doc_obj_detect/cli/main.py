"""Central CLI dispatch for training, evaluation, and utilities."""

from __future__ import annotations

import argparse
import logging

from doc_obj_detect.data.datasets import DatasetLoader
from doc_obj_detect.training import DistillRunner, EvaluatorRunner, TrainerRunner
from doc_obj_detect.visualize import visualize_augmentations

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(prog="doc-obj-detect", description="Unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_distill_parser(subparsers)
    _add_visualize_parser(subparsers)
    _add_dataset_info_parser(subparsers)

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


if __name__ == "__main__":  # pragma: no cover
    main()
