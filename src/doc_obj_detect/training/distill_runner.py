"""Runner for knowledge distillation training."""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import DFineForObjectDetection, EarlyStoppingCallback, TrainingArguments

from doc_obj_detect.config import DistillConfig, load_distill_config
from doc_obj_detect.data import collate_fn
from doc_obj_detect.models import ModelFactory
from doc_obj_detect.training.base_runner import BaseRunner, ProcessorBundle
from doc_obj_detect.training.distillation import DistillationTrainer

logger = logging.getLogger(__name__)


class DistillRunner(BaseRunner):
    """Train a student model using a frozen teacher checkpoint."""

    def __init__(self, config: DistillConfig, config_path: str | Path | None = None) -> None:
        super().__init__(config, config_path)

    @classmethod
    def from_config(cls, config_path: str | Path) -> DistillRunner:
        cfg = load_distill_config(config_path)
        return cls(cfg, config_path)

    def run(self) -> None:
        run_paths = self._prepare_run_paths()

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
        logger.info("Distillation complete. Student model saved to %s", final_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_student_model(self):
        dfine_cfg = self.config.dfine.model_dump()
        factory = ModelFactory.from_config(
            self.config.model, dfine_cfg, self.config.data.image_size
        )
        artifacts = factory.build()
        processors = self._prepare_processors(artifacts.processor)
        return artifacts.model, processors

    def _build_datasets(self, processors: ProcessorBundle):
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, class_labels = super()._build_datasets(processors)
        logger.info("Train samples: %s", len(train_dataset))
        logger.info("Val samples: %s", len(val_dataset))
        return train_dataset, val_dataset, class_labels

    def _build_training_args(self, run_paths):
        training_config_dict = self.config.training.model_dump()
        early_stopping_patience = training_config_dict.pop("early_stopping_patience", None)
        training_args = TrainingArguments(
            output_dir=str(run_paths.output_dir),
            run_name=run_paths.run_name,
            logging_dir=str(run_paths.log_dir),
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
        return training_args, callbacks


__all__ = ["DistillRunner"]
