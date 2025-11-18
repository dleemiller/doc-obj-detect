"""High-level training runner that orchestrates the HF Trainer workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainingArguments

from doc_obj_detect.config import TrainConfig, load_train_config
from doc_obj_detect.data import collate_fn
from doc_obj_detect.models import ModelFactory, get_trainable_parameters
from doc_obj_detect.training.base_runner import BaseRunner, ProcessorBundle
from doc_obj_detect.training.callbacks import UnfreezeBackboneCallback
from doc_obj_detect.training.trainer_core import SplitLRTrainer
from doc_obj_detect.utils import RunPaths

logger = logging.getLogger(__name__)


class TrainerRunner(BaseRunner):
    """Encapsulates the previous ``train.py`` script as a reusable class."""

    def __init__(self, config: TrainConfig, config_path: str | Path | None = None) -> None:
        super().__init__(config, config_path)

    @classmethod
    def from_config(cls, config_path: str | Path) -> TrainerRunner:
        cfg = load_train_config(config_path)
        return cls(cfg, config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("=" * 80)
        logger.info("Training Configuration")
        logger.info("=" * 80)

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

        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)

        trainer.train()

        final_model_path = run_paths.final_model_dir
        logger.info("Saving final model to %s", final_model_path)
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_model_path))
        processors.eval.save_pretrained(str(final_model_path))
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_model_and_processors(self) -> tuple[torch.nn.Module, ProcessorBundle]:
        model_cfg = self.config.model
        dfine_cfg = self.config.dfine.model_dump()
        image_size = self.config.data.image_size

        logger.info("Initializing model...")
        factory = ModelFactory.from_config(model_cfg, dfine_cfg, image_size=image_size)
        artifacts = factory.build()
        model = artifacts.model
        processor = artifacts.processor

        param_info = get_trainable_parameters(model)
        logger.info("Total parameters: %s", f"{param_info['total']:,}")
        logger.info("Trainable parameters: %s", f"{param_info['trainable']:,}")
        logger.info("Frozen parameters: %s", f"{param_info['frozen']:,}")
        logger.info("Trainable: %.2f%%", param_info["trainable_percent"])

        if model_cfg.pretrained_checkpoint:
            checkpoint_path = model_cfg.pretrained_checkpoint
            logger.info("Loading pretrained checkpoint: %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)

        processors = self._prepare_processors(processor)
        return model, processors

    def _build_datasets(self, processors: ProcessorBundle):
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, class_labels = super()._build_datasets(processors)
        logger.info("Train samples: %s", len(train_dataset))
        logger.info("Val samples: %s", len(val_dataset))
        logger.info("Classes: %s", class_labels)
        return train_dataset, val_dataset, class_labels

    def _build_training_args(self, model) -> tuple[TrainingArguments, list, RunPaths]:
        paths = self._prepare_run_paths()
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


__all__ = ["TrainerRunner"]
