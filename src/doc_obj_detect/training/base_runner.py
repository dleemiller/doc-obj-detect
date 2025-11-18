"""Shared utilities for training-style runners."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        if self._aug_config and self._aug_config.get("multi_scale_sizes"):
            shortest = max(self._aug_config["multi_scale_sizes"])
        else:
            shortest = self.config.data.image_size
        size = {"shortest_edge": shortest}
        if self._aug_config and self._aug_config.get("max_long_side"):
            size["longest_edge"] = self._aug_config["max_long_side"]
        return size

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
        return run_paths


__all__ = ["BaseRunner", "ProcessorBundle", "collate_fn"]
