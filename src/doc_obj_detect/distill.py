"""Knowledge distillation training entry point."""

import argparse

from doc_obj_detect.training import DistillRunner


def run(config_path: str) -> None:
    runner = DistillRunner.from_config(config_path)
    runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a student via knowledge distillation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to distillation configuration YAML file",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
