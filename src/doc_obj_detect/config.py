"""Configuration schemas using Pydantic for validation."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model configuration."""

    backbone: str = Field(description="Backbone model name from timm")
    num_classes: int = Field(gt=0, description="Number of object detection classes")
    freeze_backbone: bool = Field(
        default=False, description="Whether to freeze backbone parameters"
    )
    use_pretrained_backbone: bool = Field(
        default=True, description="Use pretrained backbone weights"
    )
    pretrained_checkpoint: str | None = Field(
        default=None, description="Path to pretrained checkpoint"
    )


class DetrConfig(BaseModel):
    """Deformable DETR architecture configuration."""

    num_queries: int = Field(default=300, gt=0, description="Number of object queries")
    encoder_layers: int = Field(default=6, gt=0, description="Number of encoder layers")
    decoder_layers: int = Field(default=6, gt=0, description="Number of decoder layers")
    encoder_attention_heads: int = Field(default=8, gt=0, description="Encoder attention heads")
    decoder_attention_heads: int = Field(default=8, gt=0, description="Decoder attention heads")
    encoder_ffn_dim: int = Field(default=1024, gt=0, description="Encoder FFN dimension")
    decoder_ffn_dim: int = Field(default=1024, gt=0, description="Decoder FFN dimension")
    num_feature_levels: int = Field(default=4, gt=0, description="Number of feature pyramid levels")
    decoder_n_points: int = Field(
        default=4, gt=0, description="Decoder deformable attention points"
    )
    encoder_n_points: int = Field(
        default=4, gt=0, description="Encoder deformable attention points"
    )

    model_config = {"extra": "allow"}  # Allow additional DETR config params


class DFineConfig(BaseModel):
    """D-FINE architecture configuration."""

    # Backbone config
    encoder_in_channels: list[int] = Field(
        default=[384, 768, 1536], description="Backbone output channels for each level"
    )
    feat_strides: list[int] = Field(default=[8, 16, 32], description="Feature strides")
    num_feature_levels: int = Field(default=3, gt=0, description="Number of feature levels")
    backbone_kwargs: dict = Field(
        default_factory=lambda: {"out_indices": (1, 2, 3)},
        description="Backbone keyword arguments",
    )

    # Encoder config
    encoder_hidden_dim: int = Field(default=256, gt=0, description="Encoder hidden dimension")
    encoder_layers: int = Field(default=1, gt=0, description="Number of encoder layers")
    encoder_ffn_dim: int = Field(default=1024, gt=0, description="Encoder FFN dimension")
    encoder_attention_heads: int = Field(default=8, gt=0, description="Encoder attention heads")

    # Decoder config
    d_model: int = Field(default=256, gt=0, description="Decoder model dimension")
    num_queries: int = Field(default=300, gt=0, description="Number of object queries")
    decoder_layers: int = Field(default=6, gt=0, description="Number of decoder layers")
    decoder_ffn_dim: int = Field(default=1024, gt=0, description="Decoder FFN dimension")
    decoder_attention_heads: int = Field(default=8, gt=0, description="Decoder attention heads")
    decoder_n_points: int = Field(default=4, gt=0, description="Deformable attention points")

    # Loss weights
    weight_loss_vfl: float = Field(default=1.0, ge=0, description="VFL loss weight")
    weight_loss_bbox: float = Field(default=5.0, ge=0, description="BBox L1 loss weight")
    weight_loss_giou: float = Field(default=2.0, ge=0, description="GIoU loss weight")
    weight_loss_fgl: float = Field(default=0.15, ge=0, description="FGL loss weight")
    weight_loss_ddf: float = Field(default=1.5, ge=0, description="DDF loss weight")

    # Training config
    num_denoising: int = Field(default=100, ge=0, description="Number of denoising queries")
    auxiliary_loss: bool = Field(default=True, description="Use auxiliary decoding losses")

    model_config = {"extra": "allow"}  # Allow additional D-FINE config params


class DataConfig(BaseModel):
    """Data configuration."""

    dataset: str = Field(description="Dataset name: 'publaynet' or 'doclaynet'")
    train_split: str = Field(default="train", description="Training data split")
    val_split: str = Field(default="val", description="Validation data split")
    test_split: str | None = Field(default=None, description="Test data split")
    image_size: int = Field(default=512, gt=0, description="Input image size")
    batch_size: int = Field(default=8, gt=0, description="Batch size per device")
    num_workers: int = Field(default=4, ge=0, description="Number of data loader workers")
    cache_dir: str | None = Field(default=None, description="Dataset cache directory")

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v: str) -> str:
        """Validate dataset name."""
        if v.lower() not in ["publaynet", "doclaynet"]:
            raise ValueError("Dataset must be 'publaynet' or 'doclaynet'")
        return v.lower()


class AugmentationConfig(BaseModel):
    """Augmentation configuration."""

    horizontal_flip: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Horizontal flip probability"
    )
    rotate_limit: int = Field(default=5, ge=0, le=90, description="Rotation angle limit in degrees")
    brightness_contrast: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Brightness/contrast variation"
    )
    noise_std: float = Field(default=0.01, ge=0.0, description="Gaussian noise standard deviation")

    model_config = {"extra": "allow"}  # Allow additional augmentation params


class TrainingConfig(BaseModel):
    """Training configuration - maps to HuggingFace TrainingArguments."""

    num_train_epochs: int = Field(gt=0, description="Number of training epochs")
    learning_rate: float = Field(gt=0, description="Learning rate")
    weight_decay: float = Field(ge=0, description="Weight decay")
    warmup_steps: int = Field(ge=0, description="Warmup steps")
    gradient_accumulation_steps: int = Field(
        default=1, gt=0, description="Gradient accumulation steps"
    )
    bf16: bool = Field(default=False, description="Use bfloat16 mixed precision")
    fp16: bool = Field(default=False, description="Use float16 mixed precision")
    save_steps: int = Field(gt=0, description="Save checkpoint every N steps")
    eval_steps: int = Field(gt=0, description="Evaluate every N steps")
    logging_steps: int = Field(gt=0, description="Log every N steps")
    eval_strategy: str = Field(default="steps", description="Evaluation strategy")
    save_strategy: str = Field(default="steps", description="Save strategy")
    save_total_limit: int = Field(default=3, gt=0, description="Maximum checkpoints to keep")
    load_best_model_at_end: bool = Field(default=True, description="Load best model at end")
    metric_for_best_model: str = Field(default="eval_loss", description="Metric for best model")
    greater_is_better: bool = Field(default=False, description="Whether higher metric is better")
    push_to_hub: bool = Field(default=False, description="Push model to HuggingFace Hub")

    model_config = {"extra": "allow"}  # Allow additional TrainingArguments params


class OutputConfig(BaseModel):
    """Output configuration."""

    output_dir: str = Field(description="Output directory for training artifacts")
    checkpoint_dir: str | None = Field(default=None, description="Checkpoint directory")
    log_dir: str | None = Field(default=None, description="TensorBoard log directory")


class TrainConfig(BaseModel):
    """Complete training configuration (D-FINE architecture)."""

    model: ModelConfig
    dfine: DFineConfig
    data: DataConfig
    augmentation: AugmentationConfig | None = None
    training: TrainingConfig
    output: OutputConfig


class DistillationConfig(BaseModel):
    """Distillation-specific configuration."""

    loss_type: str = Field(description="Distillation loss type: 'kl' or 'mse'")
    temperature: float = Field(default=3.0, gt=0, description="Temperature for KL divergence")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for distillation loss")
    beta: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for ground truth loss")
    distill_features: bool = Field(default=False, description="Distill intermediate features")

    @field_validator("loss_type")
    @classmethod
    def validate_loss_type(cls, v: str) -> str:
        """Validate loss type."""
        if v.lower() not in ["kl", "mse"]:
            raise ValueError("Loss type must be 'kl' or 'mse'")
        return v.lower()

    @field_validator("beta")
    @classmethod
    def validate_alpha_beta_sum(cls, v: float, info) -> float:
        """Validate that alpha + beta = 1.0."""
        if "alpha" in info.data:
            alpha = info.data["alpha"]
            if not abs(alpha + v - 1.0) < 1e-6:
                raise ValueError(f"alpha ({alpha}) + beta ({v}) must equal 1.0")
        return v


class TeacherConfig(BaseModel):
    """Teacher model configuration for distillation."""

    checkpoint: str = Field(description="Path to teacher model checkpoint")
    backbone: str = Field(description="Teacher backbone model name")
    detector: str = Field(description="Teacher detector type")


class StudentConfig(BaseModel):
    """Student model configuration for distillation."""

    backbone: str = Field(description="Student backbone model name")
    detector: str = Field(description="Student detector type")
    num_classes: int = Field(gt=0, description="Number of classes")


class DistillConfig(BaseModel):
    """Complete distillation configuration."""

    teacher: TeacherConfig
    student: StudentConfig
    distillation: DistillationConfig
    data: DataConfig
    augmentation: AugmentationConfig | None = None
    training: TrainingConfig
    output: OutputConfig


def load_train_config(config_path: str | Path) -> TrainConfig:
    """Load and validate training configuration.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated TrainConfig object
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return TrainConfig(**config_dict)


def load_distill_config(config_path: str | Path) -> DistillConfig:
    """Load and validate distillation configuration.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated DistillConfig object
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return DistillConfig(**config_dict)
