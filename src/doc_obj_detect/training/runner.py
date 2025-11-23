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
from doc_obj_detect.training.callbacks import EMACallback, UnfreezeBackboneCallback
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
    def run(
        self,
        resume_from_checkpoint: str | Path | None = None,
        load_weights_from: str | Path | None = None,
    ) -> None:
        """Run training with one of three modes:

        1. Fresh start (default): Build model from config
        2. Resume training (--resume): Load full state (model, optimizer, scheduler, step)
        3. Load weights (--load): Load model weights only, fresh optimizer/scheduler

        Args:
            resume_from_checkpoint: Checkpoint to resume from (full state)
            load_weights_from: Checkpoint to load weights from (weights only)
        """
        logger.info("=" * 80)
        logger.info("Training Configuration")
        logger.info("=" * 80)

        model, processors = self._build_model_and_processors(load_weights_from=load_weights_from)
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
            backbone_lr_multiplier=self.backbone_lr_multiplier,
            backbone_max_grad_norm=self.backbone_max_grad_norm,
            head_max_grad_norm=self.head_max_grad_norm,
        )

        logger.info("=" * 80)
        logger.info("Starting training...")
        if resume_from_checkpoint:
            logger.info("Mode: Resume training (full state)")
            logger.info("Resuming from checkpoint: %s", resume_from_checkpoint)
        elif load_weights_from:
            logger.info("Mode: Load weights only (fresh optimizer/scheduler)")
            logger.info("Loaded weights from: %s", load_weights_from)
        else:
            logger.info("Mode: Fresh start from config")
        logger.info("=" * 80)

        trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
        )

        # Save final model (training weights)
        final_model_path = run_paths.final_model_dir
        logger.info("Saving final model to %s", final_model_path)
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_model_path))
        processors.eval.save_pretrained(str(final_model_path))

        # Save EMA model if enabled (for better phase transitions and deployment)
        if self.config.training.ema.enabled:
            ema_callback = None
            for callback in callbacks:
                if isinstance(callback, EMACallback):
                    ema_callback = callback
                    break

            if ema_callback is not None and ema_callback.ema is not None:
                ema_model_path = run_paths.output_dir / "final_model_ema"
                logger.info("Saving EMA model to %s", ema_model_path)
                ema_model_path.mkdir(parents=True, exist_ok=True)

                # Temporarily swap to save EMA weights in HuggingFace format
                original_model = trainer.model
                trainer.model = ema_callback.ema.module
                trainer.save_model(str(ema_model_path))
                processors.eval.save_pretrained(str(ema_model_path))
                trainer.model = original_model

                logger.info(
                    "EMA model saved successfully. Use this for next phase or deployment:\n"
                    "  %s/pytorch_model.bin",
                    ema_model_path,
                )
            else:
                logger.warning(
                    "EMA enabled in config but EMA callback not found or not initialized. "
                    "Only training weights saved."
                )

        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_model_and_processors(
        self, load_weights_from: str | Path | None = None
    ) -> tuple[torch.nn.Module, ProcessorBundle]:
        """Build model and processors.

        Args:
            load_weights_from: Optional checkpoint to load weights from (--load flag)

        Returns:
            Model and processor bundle
        """
        model_cfg = self.config.model
        dfine_cfg = self.config.dfine.model_dump()
        image_size = self.config.data.image_size

        # Get class labels before building model
        class_labels = self._get_class_labels()
        logger.info("Class labels for %s: %s", self.config.data.dataset, class_labels)

        logger.info("Initializing model...")
        logger.info("dfine_cfg num_feature_levels: %s", dfine_cfg.get("num_feature_levels"))
        logger.info("dfine_cfg encoder_in_channels: %s", dfine_cfg.get("encoder_in_channels"))

        factory = ModelFactory.from_config(
            model_cfg, dfine_cfg, image_size=image_size, id2label=class_labels
        )
        artifacts = factory.build()
        model = artifacts.model
        processor = artifacts.processor

        # Load weights if --load flag provided
        if load_weights_from:
            ModelFactory.load_from_checkpoint(model, load_weights_from)

        param_info = get_trainable_parameters(model)
        logger.info("Total parameters: %s", f"{param_info['total']:,}")
        logger.info("Trainable parameters: %s", f"{param_info['trainable']:,}")
        logger.info("Frozen parameters: %s", f"{param_info['frozen']:,}")
        logger.info("Trainable: %.2f%%", param_info["trainable_percent"])

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

        # Remove custom parameters that aren't standard TrainingArguments
        early_stopping_patience = training_config_dict.pop("early_stopping_patience", None)
        _ = training_config_dict.pop("ema", None)  # EMA config
        self.backbone_lr_multiplier = training_config_dict.pop(
            "backbone_lr_multiplier", 0.01
        )  # Backbone LR multiplier
        self.backbone_max_grad_norm = training_config_dict.pop("backbone_max_grad_norm", None)
        self.head_max_grad_norm = training_config_dict.pop("head_max_grad_norm", None)

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

        # Add EMA callback if enabled
        if self.config.training.ema.enabled:
            logger.info(
                "EMA enabled: decay=%.4f, warmup_steps=%d, use_for_eval=%s",
                self.config.training.ema.decay,
                self.config.training.ema.warmup_steps,
                self.config.training.ema.use_for_eval,
            )
            callbacks.append(
                EMACallback(
                    decay=self.config.training.ema.decay,
                    warmup_steps=self.config.training.ema.warmup_steps,
                    use_ema_for_eval=self.config.training.ema.use_for_eval,
                )
            )

        # Handle backbone freezing: either freeze_backbone or freeze_backbone_epochs
        if self.config.model.freeze_backbone_epochs is not None:
            # Calculate steps per epoch: need train dataset size / batch_size
            # We'll calculate this dynamically in the trainer
            freeze_epochs = self.config.model.freeze_backbone_epochs
            if freeze_epochs > 0:
                callbacks.append(
                    UnfreezeBackboneCallback(
                        unfreeze_at_epoch=freeze_epochs  # Will be converted to steps by callback
                    )
                )
        elif self.config.model.freeze_backbone:
            callbacks.append(
                UnfreezeBackboneCallback(
                    unfreeze_at_step=training_config_dict.get("warmup_steps", 0)
                )
            )

        return training_args, callbacks, paths


__all__ = ["TrainerRunner"]
