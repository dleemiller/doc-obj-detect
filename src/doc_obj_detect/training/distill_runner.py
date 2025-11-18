"""Runner for knowledge distillation training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import DFineForObjectDetection, EarlyStoppingCallback, TrainingArguments

from doc_obj_detect.config import DistillConfig, load_distill_config
from doc_obj_detect.data import DatasetFactory, collate_fn
from doc_obj_detect.metrics import compute_map
from doc_obj_detect.models import ModelFactory
from doc_obj_detect.training.distillation import DistillationTrainer
from doc_obj_detect.utils import prepare_run_dirs, setup_logging


@dataclass
class ProcessorBundle:
    train: Any
    eval: Any


class DistillRunner:
    """Train a student model using a frozen teacher checkpoint."""

    def __init__(self, config: DistillConfig, config_path: str | Path | None = None) -> None:
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self._aug_config = config.augmentation.model_dump() if config.augmentation else None
        self._detector_stride = max(config.dfine.feat_strides)

    @classmethod
    def from_config(cls, config_path: str | Path) -> DistillRunner:
        cfg = load_distill_config(config_path)
        return cls(cfg, config_path)

    def run(self) -> None:
        run_paths = prepare_run_dirs(self.config.output)
        setup_logging(run_paths.run_name, run_paths.log_dir)

        student_model, processors = self._build_student_model()
        teacher_model = DFineForObjectDetection.from_pretrained(self.config.teacher.checkpoint)

        train_dataset, val_dataset, class_labels = self._build_datasets(processors)
        training_args, callbacks = self._build_training_args(run_paths)

        compute_metrics_fn = self._build_metrics_fn(processors.eval, class_labels)

        trainer = DistillationTrainer(
            model=student_model,
            teacher_model=teacher_model,
            distill_config=self.config.distillation,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            processing_class=processors.train,
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
        )

        trainer.train()
        final_dir = run_paths.final_model_dir
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        processors.eval.save_pretrained(str(final_dir))
        print(f"Distillation complete. Student model saved to {final_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_student_model(self):
        dfine_cfg = self.config.dfine.model_dump()
        factory = ModelFactory.from_config(
            self.config.model, dfine_cfg, self.config.data.image_size
        )
        artifacts = factory.build()
        processor = artifacts.processor

        train_processor = processor
        train_processor.do_resize = False
        train_processor.do_pad = True

        eval_processor = copy.deepcopy(processor)
        eval_processor.do_resize = True
        eval_processor.do_pad = True
        eval_processor.size = self._build_eval_size()

        return artifacts.model, ProcessorBundle(train=train_processor, eval=eval_processor)

    def _build_eval_size(self):
        if self._aug_config and self._aug_config.get("multi_scale_sizes"):
            shortest = max(self._aug_config["multi_scale_sizes"])
        else:
            shortest = self.config.data.image_size
        size = {"shortest_edge": shortest}
        if self._aug_config and self._aug_config.get("max_long_side"):
            size["longest_edge"] = self._aug_config["max_long_side"]
        return size

    def _build_datasets(self, processors):
        data_cfg = self.config.data
        train_factory = DatasetFactory(
            dataset_name=data_cfg.dataset,
            image_processor=processors.train,
            pad_stride=self._detector_stride,
            cache_dir=data_cfg.cache_dir,
            augmentation_config=self._aug_config,
        )
        train_dataset, _ = train_factory.build(split=data_cfg.train_split, apply_augmentation=True)

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
        return train_dataset, val_dataset, class_labels

    def _build_training_args(self, run_paths):
        data_cfg = self.config.data
        training_config_dict = self.config.training.model_dump()
        early_stopping_patience = training_config_dict.pop("early_stopping_patience", None)
        training_args = TrainingArguments(
            output_dir=str(run_paths.output_dir),
            run_name=run_paths.run_name,
            logging_dir=str(run_paths.log_dir),
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_num_workers=data_cfg.num_workers,
            per_device_train_batch_size=data_cfg.batch_size,
            per_device_eval_batch_size=data_cfg.batch_size,
            **training_config_dict,
        )
        callbacks = []
        if early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        return training_args, callbacks

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


__all__ = ["DistillRunner"]
