"""High-level training runner that orchestrates the HF Trainer workflow."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import EarlyStoppingCallback, TrainingArguments

from doc_obj_detect.config import TrainConfig, load_train_config
from doc_obj_detect.data import DatasetFactory, collate_fn
from doc_obj_detect.metrics import compute_map
from doc_obj_detect.models import ModelFactory, get_trainable_parameters
from doc_obj_detect.training.callbacks import UnfreezeBackboneCallback
from doc_obj_detect.training.trainer_core import SplitLRTrainer
from doc_obj_detect.utils import RunPaths, prepare_run_dirs, setup_logging


@dataclass
class ProcessorBundle:
    """Container for training/evaluation image processors."""

    train: Any
    eval: Any


class TrainerRunner:
    """Encapsulates the previous ``train.py`` script as a reusable class."""

    def __init__(self, config: TrainConfig, config_path: str | Path | None = None) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self._aug_config = config.augmentation.model_dump() if config.augmentation else None
        self._detector_stride = max(config.dfine.feat_strides)

    @classmethod
    def from_config(cls, config_path: str | Path) -> TrainerRunner:
        cfg = load_train_config(config_path)
        return cls(cfg, config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("=" * 80)
        print("Training Configuration")
        print("=" * 80)

        model, processors = self._build_model_and_processors()
        train_dataset, val_dataset, class_labels = self._build_datasets(processors)
        training_args, callbacks, run_paths = self._build_training_args(model)

        compute_metrics_fn = self._build_metrics_fn(processors.eval, class_labels)

        trainer = SplitLRTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            processing_class=processors.train,
            callbacks=callbacks,
            compute_metrics=compute_metrics_fn,
        )

        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

        trainer.train()

        final_model_path = run_paths.final_model_dir
        print(f"\nSaving final model to {final_model_path}...")
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_model_path))
        processors.eval.save_pretrained(str(final_model_path))
        print("Training complete.")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_model_and_processors(self) -> tuple[torch.nn.Module, ProcessorBundle]:
        model_cfg = self.config.model
        dfine_cfg = self.config.dfine.model_dump()
        image_size = self.config.data.image_size

        print("\nInitializing model...")
        factory = ModelFactory.from_config(model_cfg, dfine_cfg, image_size=image_size)
        artifacts = factory.build()
        model = artifacts.model
        processor = artifacts.processor

        param_info = get_trainable_parameters(model)
        print(f"Total parameters: {param_info['total']:,}")
        print(f"Trainable parameters: {param_info['trainable']:,}")
        print(f"Frozen parameters: {param_info['frozen']:,}")
        print(f"Trainable: {param_info['trainable_percent']:.2f}%")

        if model_cfg.pretrained_checkpoint:
            checkpoint_path = model_cfg.pretrained_checkpoint
            print(f"\nLoading pretrained checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)

        train_processor = processor
        train_processor.do_resize = False
        train_processor.do_pad = True

        eval_processor = copy.deepcopy(processor)
        eval_processor.do_resize = True
        eval_processor.do_pad = True
        eval_processor.size = self._build_eval_size()

        return model, ProcessorBundle(train=train_processor, eval=eval_processor)

    def _build_eval_size(self) -> dict:
        if self._aug_config and self._aug_config.get("multi_scale_sizes"):
            eval_short_side = max(self._aug_config["multi_scale_sizes"])
        else:
            eval_short_side = self.config.data.image_size
        eval_size = {"shortest_edge": eval_short_side}
        if self._aug_config and self._aug_config.get("max_long_side"):
            eval_size["longest_edge"] = self._aug_config["max_long_side"]
        return eval_size

    def _build_datasets(self, processors: ProcessorBundle):
        data_cfg = self.config.data

        print("\nPreparing datasets...")
        train_factory = DatasetFactory(
            dataset_name=data_cfg.dataset,
            image_processor=processors.train,
            pad_stride=self._detector_stride,
            cache_dir=data_cfg.cache_dir,
            augmentation_config=self._aug_config,
        )
        train_dataset, _ = train_factory.build(
            split=data_cfg.train_split,
            apply_augmentation=True,
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
            max_samples=data_cfg.max_eval_samples,
            apply_augmentation=False,
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Classes: {class_labels}")

        return train_dataset, val_dataset, class_labels

    def _build_training_args(self, model) -> tuple[TrainingArguments, list, RunPaths]:
        paths = prepare_run_dirs(self.config.output)
        setup_logging(paths.run_name, paths.log_dir)
        print(f"TensorBoard run name: {paths.run_name}")
        print(f"Logs will be saved to: {paths.log_dir}")
        training_config_dict = self.config.training.model_dump()
        early_stopping_patience = training_config_dict.pop("early_stopping_patience", None)

        training_args = TrainingArguments(
            output_dir=str(paths.output_dir),
            run_name=paths.run_name,
            logging_dir=str(paths.log_dir),
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_num_workers=self.config.data.num_workers,
            per_device_train_batch_size=self.config.data.batch_size,
            per_device_eval_batch_size=self.config.data.batch_size,
            **training_config_dict,
        )

        callbacks = []
        if early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        if self.config.model.freeze_backbone:
            callbacks.append(
                UnfreezeBackboneCallback(
                    unfreeze_at_step=training_config_dict.get("warmup_steps", 0)
                )
            )

        return training_args, callbacks, paths

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


__all__ = ["TrainerRunner"]
