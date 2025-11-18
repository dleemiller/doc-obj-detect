"""Thin CLI wrapper around :class:`TrainerRunner`."""

import argparse

from doc_obj_detect.training import TrainerRunner


def train(config_path: str) -> None:
    """Train object detection model using the new runner."""
    runner = TrainerRunner.from_config(config_path)
    runner.run()


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train document object detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
