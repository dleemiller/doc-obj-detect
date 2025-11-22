"""Configuration schemas using Pydantic for validation."""

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Model configuration."""

    backbone: str = Field(description="Backbone model name from timm")
    num_classes: int = Field(gt=0, description="Number of object detection classes")
    freeze_backbone: bool = Field(
        default=False, description="Whether to freeze backbone parameters"
    )
    freeze_backbone_epochs: int | None = Field(
        default=None,
        ge=0,
        description="Number of epochs to freeze backbone (None = use freeze_backbone)",
    )
    use_pretrained_backbone: bool = Field(
        default=True, description="Use pretrained backbone weights"
    )
    use_pretrained_head: bool = Field(
        default=True, description="Use pretrained D-FINE head weights (False = train from scratch)"
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
    decoder_n_points: int | list[int] = Field(
        default=[3, 6, 3], description="Decoder deformable attention points (int or list per level)"
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
    decoder_n_points: int | list[int] = Field(
        default=[3, 6, 3], description="Deformable attention points (int or list per level)"
    )

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
    image_size: int = Field(
        default=512, ge=64, le=4096, description="Input image size (short side for square mode)"
    )
    batch_size: int = Field(default=8, gt=0, le=1024, description="Batch size per device")
    num_workers: int = Field(
        default=4, ge=0, le=128, description="Number of data loader workers (total per process)"
    )
    cache_dir: str | None = Field(default=None, description="Dataset cache directory")
    max_eval_samples: int = Field(
        default=2000, ge=0, le=200_000, description="Maximum samples to evaluate per run"
    )

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v: str) -> str:
        """Validate dataset name."""
        if v.lower() not in ["publaynet", "doclaynet"]:
            raise ValueError("Dataset must be 'publaynet' or 'doclaynet'")
        return v.lower()


class PerspectiveAugConfig(BaseModel):
    probability: float = Field(
        default=0.3, ge=0, le=1, description="Probability of perspective warp"
    )
    scale_min: float = Field(default=0.02, ge=0, description="Minimum perspective scale")
    scale_max: float = Field(default=0.05, ge=0, description="Maximum perspective scale")


class ElasticAugConfig(BaseModel):
    probability: float = Field(
        default=0.2, ge=0, le=1, description="Probability of elastic transform"
    )
    alpha: float = Field(default=30, ge=0, description="Elastic alpha magnitude")
    sigma: float = Field(default=5, ge=0, description="Elastic sigma")


class BrightnessContrastConfig(BaseModel):
    limit: float = Field(default=0.2, ge=0, description="Brightness/contrast variation limit")
    probability: float = Field(
        default=0.5, ge=0, le=1, description="Probability of brightness/contrast change"
    )


class BlurConfig(BaseModel):
    probability: float = Field(
        default=0.3, ge=0, le=1, description="Probability of blur augmentations"
    )
    blur_limit: int = Field(default=3, ge=1, description="Maximum blur kernel size")


class CompressionConfig(BaseModel):
    probability: float = Field(
        default=0.3, ge=0, le=1, description="Probability of JPEG compression"
    )
    quality_min: int = Field(default=75, ge=1, description="Minimum JPEG quality")
    quality_max: int = Field(default=100, ge=1, description="Maximum JPEG quality")


class NoiseConfig(BaseModel):
    probability: float = Field(default=0.2, ge=0, le=1, description="Probability of Gaussian noise")
    std_min: float = Field(default=0.0, ge=0, description="Minimum noise standard deviation")
    std_max: float = Field(default=0.01, ge=0, description="Maximum noise standard deviation")


class AugmentationConfig(BaseModel):
    """Augmentation configuration."""

    multi_scale_sizes: list[int] = Field(
        default_factory=lambda: [512],
        description="Desired short-side sizes for multi-scale training.",
    )
    max_long_side: int | None = Field(
        default=960,
        ge=64,
        le=8192,
        description="Maximum allowed long-side size when aspect-preserving resize is enabled.",
    )
    force_square_resize: bool = Field(
        default=False,
        description="Use legacy square resizing instead of aspect-preserving multi-scale.",
    )
    rotate_limit: int = Field(default=5, ge=0, le=90, description="Rotation angle limit in degrees")
    rotate_prob: float = Field(
        default=0.5, ge=0, le=1, description="Probability of rotation augmentation"
    )
    perspective: PerspectiveAugConfig = Field(default_factory=PerspectiveAugConfig)
    elastic: ElasticAugConfig = Field(default_factory=ElasticAugConfig)
    brightness_contrast: BrightnessContrastConfig = Field(default_factory=BrightnessContrastConfig)
    blur: BlurConfig = Field(default_factory=BlurConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)

    model_config = {"extra": "allow"}  # Allow additional augmentation params

    @field_validator("multi_scale_sizes")
    @classmethod
    def validate_multi_scale_sizes(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("multi_scale_sizes must contain at least one size.")
        for size in v:
            if size < 64 or size > 4096:
                raise ValueError("multi_scale_sizes entries must be between 64 and 4096 pixels.")
        return v

    @model_validator(mode="after")
    def ensure_long_side(self) -> "AugmentationConfig":
        max_scale = max(self.multi_scale_sizes) if self.multi_scale_sizes else 0
        if self.max_long_side is not None and self.max_long_side < max_scale:
            raise ValueError("max_long_side must be >= largest entry in multi_scale_sizes.")
        return self


class EMAConfig(BaseModel):
    """Exponential Moving Average configuration."""

    enabled: bool = Field(default=False, description="Enable EMA for model weights")
    decay: float = Field(
        default=0.9999, ge=0, le=1, description="EMA decay rate (D-FINE default: 0.9999)"
    )
    warmup_steps: int = Field(
        default=1000, ge=0, description="EMA warmup steps (D-FINE default: 1000)"
    )
    use_for_eval: bool = Field(
        default=True, description="Use EMA weights for evaluation (recommended)"
    )


class TrainingConfig(BaseModel):
    """Training configuration - maps to HuggingFace TrainingArguments."""

    num_train_epochs: int = Field(gt=0, le=1000, description="Number of training epochs")
    learning_rate: float = Field(gt=0, le=1.0, description="Learning rate")
    weight_decay: float = Field(ge=0, le=1.0, description="Weight decay")
    # warmup_steps: int | None = Field(default=None, ge=0, description="Warmup steps (optional)")
    warmup_ratio: float | None = Field(
        default=None, ge=0, le=1, description="Warmup ratio (optional, alternative to warmup_steps)"
    )
    gradient_accumulation_steps: int = Field(
        default=1, gt=0, le=8192, description="Gradient accumulation steps"
    )
    bf16: bool = Field(default=False, description="Use bfloat16 mixed precision")
    fp16: bool = Field(default=False, description="Use float16 mixed precision")
    save_steps: int = Field(gt=0, le=1_000_000, description="Save checkpoint every N steps")
    eval_steps: int = Field(gt=0, le=1_000_000, description="Evaluate every N steps")
    logging_steps: int = Field(gt=0, le=1_000_000, description="Log every N steps")
    eval_strategy: str = Field(default="steps", description="Evaluation strategy")
    save_strategy: str = Field(default="steps", description="Save strategy")
    save_total_limit: int = Field(default=3, gt=0, description="Maximum checkpoints to keep")
    load_best_model_at_end: bool = Field(default=True, description="Load best model at end")
    metric_for_best_model: str = Field(default="eval_loss", description="Metric for best model")
    greater_is_better: bool = Field(default=False, description="Whether higher metric is better")
    push_to_hub: bool = Field(default=False, description="Push model to HuggingFace Hub")
    early_stopping_patience: int | None = Field(
        default=None,
        ge=1,
        description="Optional patience for EarlyStoppingCallback",
    )
    ema: EMAConfig = Field(default_factory=EMAConfig, description="EMA configuration")

    model_config = {"extra": "allow"}  # Allow additional TrainingArguments params


class OutputConfig(BaseModel):
    """Output configuration."""

    output_dir: str = Field(description="Output directory for training artifacts")
    checkpoint_dir: str | None = Field(default=None, description="Checkpoint directory")
    log_dir: str | None = Field(default=None, description="TensorBoard log directory")
    run_name: str | None = Field(
        default=None,
        description="Unique run name for TensorBoard (creates subdirectory in log_dir). "
        "If None, uses timestamp.",
    )


class TrainConfig(BaseModel):
    """Complete training configuration (D-FINE architecture)."""

    model: ModelConfig
    dfine: DFineConfig | None = None
    detr: DFineConfig | None = None
    data: DataConfig
    augmentation: AugmentationConfig | None = None
    training: TrainingConfig
    output: OutputConfig

    @model_validator(mode="after")
    def ensure_dfine(self) -> "TrainConfig":
        dfine = self.dfine
        detr = self.detr
        if dfine is None and detr is None:
            raise ValueError("Either 'dfine' or 'detr' section must be provided in the config.")
        if dfine is None:
            self.dfine = detr
        if self.detr is None:
            self.detr = self.dfine
        return self


class DistillationConfig(BaseModel):
    """Distillation-specific configuration."""

    loss_type: str = Field(description="Distillation loss type: 'kl' or 'mse'")
    temperature: float = Field(
        default=3.0, gt=0, le=10.0, description="Temperature for KL divergence"
    )
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


class DistillConfig(BaseModel):
    """Complete distillation configuration."""

    teacher: TeacherConfig
    model: ModelConfig
    dfine: DFineConfig
    distillation: DistillationConfig
    data: DataConfig
    augmentation: AugmentationConfig | None = None
    training: TrainingConfig
    output: OutputConfig
