"""Training script for document object detection models."""

import argparse
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments

from doc_obj_detect.config import load_train_config
from doc_obj_detect.data import collate_fn, prepare_dataset_for_training
from doc_obj_detect.metrics import compute_map
from doc_obj_detect.model import create_model, get_trainable_parameters
from doc_obj_detect.trainer import SplitLRTrainer


def unfreeze_backbone(model):
    for p in model.model.backbone.parameters():
        p.requires_grad = True


class UnfreezeBackboneCallback(TrainerCallback):
    def __init__(self, unfreeze_at_step):
        self.unfreeze_at_step = unfreeze_at_step
        self.done = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.done and state.global_step >= self.unfreeze_at_step:
            model = kwargs["model"]
            for p in model.model.backbone.parameters():
                p.requires_grad = True
            print(f"[Callback] Unfroze backbone at step {state.global_step}")
            self.done = True
            return control


def train(config_path: str) -> None:
    """Train object detection model.

    Args:
        config_path: Path to training configuration YAML file
    """
    # Load and validate configuration
    config = load_train_config(config_path)

    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)

    # Create model
    print("\nInitializing model...")
    model_config = config.model
    dfine_config = config.dfine.model_dump()

    model, image_processor = create_model(
        backbone=model_config.backbone,
        num_classes=model_config.num_classes,
        use_pretrained_backbone=model_config.use_pretrained_backbone,
        freeze_backbone=model_config.freeze_backbone,
        image_size=config.data.image_size,
        **dfine_config,
    )

    # Print model info
    param_info = get_trainable_parameters(model)
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Frozen parameters: {param_info['frozen']:,}")
    print(f"Trainable: {param_info['trainable_percent']:.2f}%")

    # Load pretrained checkpoint if specified
    if model_config.pretrained_checkpoint:
        checkpoint_path = model_config.pretrained_checkpoint
        print(f"\nLoading pretrained checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)

    # Prepare datasets
    print("\nPreparing datasets...")
    data_config = config.data
    aug_config = config.augmentation.model_dump() if config.augmentation else None

    train_dataset, _ = prepare_dataset_for_training(
        dataset_name=data_config.dataset,
        split=data_config.train_split,
        image_processor=image_processor,
        augmentation_config=aug_config,
        cache_dir=data_config.cache_dir,
    )

    val_dataset, class_labels = prepare_dataset_for_training(
        dataset_name=data_config.dataset,
        split=data_config.val_split,
        image_processor=image_processor,
        augmentation_config=None,  # No augmentation for validation
        cache_dir=data_config.cache_dir,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {class_labels}")

    # Setup training arguments
    print("\nConfiguring training...")
    output_config = config.output

    # Create output directories
    output_dir = Path(output_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract early_stopping_patience from training config (not a TrainingArguments param)
    training_config_dict = config.training.model_dump()
    early_stopping_patience = training_config_dict.pop("early_stopping_patience", None)

    # Build training arguments from config
    # Note: SplitLRTrainer overrides get_eval_dataloader to use num_workers=0
    # to avoid "too many open files" when train/eval dataloaders run concurrently
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=output_config.log_dir,
        report_to=["tensorboard"],  # Use TensorBoard for monitoring
        remove_unused_columns=False,  # Keep all columns for custom collate
        dataloader_num_workers=data_config.num_workers,
        per_device_train_batch_size=data_config.batch_size,
        per_device_eval_batch_size=data_config.batch_size,
        **training_config_dict,  # All training params from config
    )

    # Setup callbacks
    callbacks = []
    # Only add unfreeze callback if backbone starts frozen
    if model_config.freeze_backbone:
        callbacks.append(
            UnfreezeBackboneCallback(unfreeze_at_step=training_config_dict["warmup_steps"])
        )
    if early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    # Create compute_metrics function with closure over image_processor and id2label
    def compute_metrics_fn(eval_pred):
        return compute_map(
            eval_pred=eval_pred,
            image_processor=image_processor,
            id2label=dict(enumerate(class_labels)),
            threshold=0.0,  # Post-process all detections, NMS happens in post_process
            max_eval_images=2000,  # Limit to 2000 images to avoid OOM
        )

    # Initialize trainer
    trainer = SplitLRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=image_processor,  # Save processor with model
        callbacks=callbacks,
        compute_metrics=compute_metrics_fn,
    )

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    # Train
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    image_processor.save_pretrained(str(final_model_path))

    print(f"\nTraining complete! Model saved to: {final_model_path}")


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
