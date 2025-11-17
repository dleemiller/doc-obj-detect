"""Command-line utilities."""

import argparse

from doc_obj_detect.data import load_doclaynet, load_publaynet
from doc_obj_detect.visualize import visualize_augmentations


def preprocess_cli() -> None:
    """Entry point for dataset inspection and visualization."""
    parser = argparse.ArgumentParser(description="Preprocess datasets for training.")
    parser.add_argument(
        "command",
        choices=["load", "visualize"],
        help="Select 'load' to verify dataset access or 'visualize' for augmentation previews.",
    )
    parser.add_argument(
        "dataset",
        choices=["publaynet", "doclaynet"],
        help="Dataset to operate on.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional datasets cache directory.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples for visualization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/augmentation_samples",
        help="Destination for visualization artifacts.",
    )
    args = parser.parse_args()

    if args.command == "load":
        _load_dataset_summary(args.dataset, args.cache_dir)
    else:
        visualize_augmentations(
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )


def _load_dataset_summary(dataset: str, cache_dir: str | None) -> None:
    """Print dataset statistics for a quick sanity check."""
    print(f"Loading {dataset} dataset...")
    if dataset == "publaynet":
        train_ds, labels = load_publaynet("train", cache_dir)
        val_ds, _ = load_publaynet("val", cache_dir)
    else:
        train_ds, labels = load_doclaynet("train", cache_dir)
        val_ds, _ = load_doclaynet("val", cache_dir)

    print("\nDataset loaded successfully!")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Classes: {labels}")
