"""Training script for document object detection models."""

import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from doc_obj_detect.config import load_train_config
from doc_obj_detect.data import collate_fn, prepare_dataset_for_training
from doc_obj_detect.model import create_model, get_trainable_parameters


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
    detr_config = config.detr.model_dump()

    model, image_processor = create_model(
        backbone=model_config.backbone,
        num_classes=model_config.num_classes,
        use_pretrained_backbone=model_config.use_pretrained_backbone,
        freeze_backbone=model_config.freeze_backbone,
        image_size=config.data.image_size,
        **detr_config,
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

    # Build training arguments from config
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=output_config.log_dir,
        report_to=["tensorboard"],  # Use TensorBoard for monitoring
        remove_unused_columns=False,  # Keep all columns for custom collate
        dataloader_num_workers=data_config.num_workers,
        per_device_train_batch_size=data_config.batch_size,
        per_device_eval_batch_size=data_config.batch_size,
        **config.training.model_dump(),  # All training params from config
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=image_processor,  # Save processor with model
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
