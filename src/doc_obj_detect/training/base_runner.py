"""Shared utilities for training-style runners."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from doc_obj_detect.data import DatasetFactory, collate_fn
from doc_obj_detect.metrics import compute_map
from doc_obj_detect.utils import RunPaths, prepare_run_dirs, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessorBundle:
    """Pair of processors for train/eval flows."""

    train: Any
    eval: Any


class BaseRunner:
    """Common helper methods shared by TrainerRunner and DistillRunner."""

    def __init__(self, config, config_path: str | Path | None = None) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self._aug_config = config.augmentation.model_dump() if config.augmentation else None
        self._detector_stride = max(config.dfine.feat_strides)

    # ------------------------------------------------------------------
    # Processor & dataset helpers
    # ------------------------------------------------------------------
    def _prepare_processors(self, processor) -> ProcessorBundle:
        train_processor = processor
        train_processor.do_resize = False
        train_processor.do_pad = True

        eval_processor = copy.deepcopy(processor)
        eval_processor.do_resize = True
        eval_processor.do_pad = True
        eval_processor.size = self._build_eval_size()

        return ProcessorBundle(train=train_processor, eval=eval_processor)

    def _build_eval_size(self) -> dict[str, int]:
        # Determine base size
        if self._aug_config and self._aug_config.get("multi_scale_sizes"):
            shortest = max(self._aug_config["multi_scale_sizes"])
        else:
            shortest = self.config.data.image_size

        # If training with square resize, eval should also use square
        if self._aug_config and self._aug_config.get("force_square_resize", False):
            return {"height": shortest, "width": shortest}

        # Otherwise use shortest_edge with optional max_long_side
        size = {"shortest_edge": shortest}
        if self._aug_config and self._aug_config.get("max_long_side"):
            size["longest_edge"] = self._aug_config["max_long_side"]
        return size

    def _get_class_labels(self) -> dict[int, str]:
        """Get class labels for the dataset without loading the full dataset."""

        dataset_name = self.config.data.dataset.lower()
        if dataset_name == "publaynet":
            from doc_obj_detect.data.constants import PUBLAYNET_CLASSES

            return PUBLAYNET_CLASSES
        elif dataset_name == "doclaynet":
            from doc_obj_detect.data.constants import DOCLAYNET_CLASSES

            return DOCLAYNET_CLASSES
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _build_datasets(
        self,
        processors: ProcessorBundle,
        max_eval_samples: int | None = None,
        apply_train_augmentation: bool = True,
    ):
        data_cfg = self.config.data

        train_factory = DatasetFactory(
            dataset_name=data_cfg.dataset,
            image_processor=processors.train,
            pad_stride=self._detector_stride,
            cache_dir=data_cfg.cache_dir,
            augmentation_config=self._aug_config if apply_train_augmentation else None,
        )
        train_dataset, _ = train_factory.build(
            split=data_cfg.train_split,
            apply_augmentation=apply_train_augmentation,
        )

        eval_factory = DatasetFactory(
            dataset_name=data_cfg.dataset,
            image_processor=processors.eval,
            pad_stride=self._detector_stride,
            cache_dir=data_cfg.cache_dir,
            augmentation_config=None,
        )
        val_dataset, class_labels = eval_factory.build(
            split=data_cfg.val_split,
            max_samples=max_eval_samples or data_cfg.max_eval_samples,
            apply_augmentation=False,
        )
        return train_dataset, val_dataset, class_labels

    def _build_metrics_fn(self, eval_processor, class_labels):
        data_cfg = self.config.data

        def compute_metrics_fn(eval_pred):
            return compute_map(
                eval_pred=eval_pred,
                image_processor=eval_processor,
                id2label=class_labels,
                threshold=0.0,
                max_eval_images=data_cfg.max_eval_samples,
            )

        return compute_metrics_fn

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------
    def _prepare_run_paths(self) -> RunPaths:
        run_paths = prepare_run_dirs(self.config.output)
        setup_logging(run_paths.run_name, run_paths.log_dir)
        logger.info("TensorBoard run name: %s", run_paths.run_name)
        logger.info("Logs will be saved to: %s", run_paths.log_dir)
        self._write_run_metadata(run_paths)
        return run_paths

    def _write_run_metadata(self, run_paths: RunPaths) -> None:
        """Persist the resolved configuration + run metadata for reproducibility."""

        metadata_root = Path(run_paths.output_dir) / "run_metadata" / run_paths.run_name
        metadata_root.mkdir(parents=True, exist_ok=True)

        config_path = metadata_root / "config.yaml"
        metadata_path = metadata_root / "meta.json"

        config_dict = self.config.model_dump(mode="json")
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_dict, handle, sort_keys=False)

        metadata = {
            "run_name": run_paths.run_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(self.config_path) if self.config_path else None,
            "output_dir": str(run_paths.output_dir),
            "log_dir": str(run_paths.log_dir),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        logger.info("Saved config snapshot to %s", config_path)
        logger.info("Saved run metadata to %s", metadata_path)


__all__ = ["BaseRunner", "ProcessorBundle", "collate_fn"]
