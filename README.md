# Document Layout Analysis with D-FINE

ConvNeXt-DINOv3 backbone with D-FINE detector for document object detection. Train on PubLayNet, fine-tune on DocLayNet, then distill to smaller variants for production deployment.

## Features

- **Teacher model:** ConvNeXt-Large DINOv3 + D-FINE with 300 queries (3-level feature pyramid)
- **Augmentation:** ICDAR 2023 competition strategies (mosaic, flips, rotation) with epoch-based scheduling
- **Student models:** Distillation to smaller ConvNeXt variants (Base/Small/Tiny) + D-FINE
- **Datasets:** PubLayNet (5 classes) pretraining, DocLayNet (11 classes) fine-tuning
- **Architecture system:** Modular D-FINE variants (XLarge/Large/Medium/Small) in separate configs
- **Testing:** 69% coverage with CPU-only tests for training components

## Quick Start

```bash
git clone git@github.com:dleemiller/doc-obj-detect.git
cd doc-obj-detect
uv sync

# Run tests
uv run pytest --cov=src

# Start training
uv run doc-obj-detect train --config configs/pretrain_publaynet_square640.yaml
```

## Training Pipeline

### Phase 1: Pretrain on PubLayNet
```bash
# Square 640x640 (recommended - simpler, faster, ICDAR baseline approach)
uv run doc-obj-detect train --config configs/pretrain_publaynet_square640.yaml

# Alternative: Aspect ratio preserving (DocLayout-YOLO approach)
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml
```

**Square config highlights:**
- 640x640 resolution with multi-scale [448, 480, 512, 544, 576, 608, 640]
- Unfrozen backbone from start (simpler training dynamics)
- Batch size 32, LR 2.5e-4
- Mosaic augmentation (50% prob, disabled after epoch 10)
- Flips: horizontal 50%, vertical 50%

### Phase 2: Fine-tune on DocLayNet
```bash
# Load pretrained weights (automatically uses EMA if available)
uv run doc-obj-detect train \
  --config configs/finetune_doclaynet.yaml \
  --load outputs/pretrain_publaynet_square640/checkpoint-5000

# Force non-EMA weights if needed
uv run doc-obj-detect train \
  --config configs/finetune_doclaynet.yaml \
  --load outputs/pretrain_publaynet_square640/checkpoint-5000 \
  --no-ema
```

**Finetuning differences:**
- 11 classes vs 5 in pretraining
- Lower base LR (5.0e-5)
- Higher backbone LR multiplier (0.1 vs 0.01)
- End-to-end training from start

### Phase 3: Distill to Smaller Models
```bash
uv run doc-obj-detect distill --config configs/distill.yaml

# Modify configs/distill.yaml to experiment:
# - model.architecture: dfine_medium / dfine_small
# - model.backbone: convnext_base.dinov2 / convnext_small
# - distillation.alpha: 0.7  # Distillation loss weight
# - distillation.beta: 0.3   # Ground truth loss weight
```

### Resume Interrupted Training
```bash
# Resume with full state (model, optimizer, scheduler, step)
uv run doc-obj-detect train \
  --config configs/pretrain_publaynet_square640.yaml \
  --resume-from-checkpoint outputs/pretrain_publaynet_square640/checkpoint-3000
```

## Augmentation Strategy

Following ICDAR 2023 competition winners:

**Spatial augmentations:**
- Mosaic (50% prob, disabled after epoch 10) - combines 4 images in 2x2 grid
- Horizontal flip (50%)
- Vertical flip (50%)
- Rotation (±5°, 50% prob)

**Photometric augmentations:**
- Brightness/contrast (±20%, 50% prob)
- Blur (motion/gaussian, 30% prob)
- JPEG compression (75-100 quality, 30% prob)
- Gaussian noise (30% prob)

**Implementation:**
- Uses official albumentations library
- Mosaic handles mixed image sizes automatically
- Epoch-based scheduling via `set_epoch()` callback

## Configuration System

### Architecture Selection
```yaml
model:
  architecture: dfine_xlarge  # or: dfine_large, dfine_medium, dfine_small
```

Architectures defined in `configs/architectures/`:
- `dfine_xlarge.yaml` - 6 decoder layers, silu activation (best accuracy)
- `dfine_large.yaml` - 6 decoder layers, relu activation
- `dfine_medium.yaml` - 4 decoder layers, depth_mult=0.67
- `dfine_small.yaml` - 3 decoder layers, depth_mult=0.34

### Training Settings
```yaml
training:
  learning_rate: 2.5e-4           # Base LR for detection head
  backbone_lr_multiplier: 0.01    # Backbone gets 1% of head LR
  max_grad_norm: 0.1              # Global gradient clipping

  # Optional: separate backbone/head gradient clipping
  # backbone_max_grad_norm: 0.1
  # head_max_grad_norm: 0.1

  # Optimizer
  optim: adamw_torch_fused
  weight_decay: 1.25e-4

  # Scheduler
  lr_scheduler_type: cosine_with_min_lr
  warmup_ratio: 0.08
  lr_scheduler_kwargs:
    min_lr: 2.5e-5

  # EMA
  ema:
    enabled: true
    decay: 0.9999
    warmup_steps: 1000
    use_for_eval: true
```

### Augmentation Configuration
```yaml
augmentation:
  force_square_resize: true              # Square 640x640
  multi_scale_sizes: [448, 512, 576, 640]

  horizontal_flip: 0.5
  vertical_flip: 0.5

  rotate_limit: 5
  rotate_prob: 0.5

  mosaic:
    probability: 0.5
    disable_after_epoch: 10

  brightness_contrast:
    limit: 0.2
    probability: 0.5
```

## CLI Commands

```bash
# Training
uv run doc-obj-detect train --config <config.yaml>
uv run doc-obj-detect train --config <config.yaml> --load <checkpoint>
uv run doc-obj-detect train --config <config.yaml> --resume-from-checkpoint <checkpoint>

# Evaluation
uv run doc-obj-detect evaluate --config <config.yaml> --checkpoint <path>

# Distillation
uv run doc-obj-detect distill --config configs/distill.yaml

# Utilities
uv run doc-obj-detect visualize --dataset publaynet
uv run doc-obj-detect dataset-info --dataset doclaynet

# Monitoring
tensorboard --logdir outputs/<run>/logs
```

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# HTML coverage report
uv run pytest --cov=src --cov-report=html  # Open htmlcov/index.html

# Specific test
uv run pytest tests/test_model_loading.py -v
```

**Test coverage:** 69% overall, 88-99% on critical components (model loading, metrics, callbacks)

## Project Structure

```
src/doc_obj_detect/
├─ cli/                 # CLI entry point and subcommands
├─ config/              # Pydantic schemas, YAML loaders
├─ data/                # Dataset wrappers, augmentation, collate
├─ models/              # Model factory, checkpoint loading
├─ training/
│   ├─ runner.py        # Main training orchestration
│   ├─ distill_runner.py# Distillation orchestration
│   ├─ trainer_core.py  # Custom HF Trainer (split LR, grad clipping)
│   ├─ callbacks.py     # EMA, backbone unfreezing
│   └─ evaluator.py     # Checkpoint evaluation
└─ metrics.py           # COCO-style mAP

configs/
├─ architectures/       # D-FINE variant definitions
├─ pretrain_publaynet.yaml
├─ pretrain_publaynet_square640.yaml
├─ finetune_doclaynet.yaml
└─ distill.yaml

tests/                  # 69% coverage
```

## Key Implementation Details

**Split learning rates:**
- Backbone gets 1-10% of head LR (configurable via `backbone_lr_multiplier`)
- Prevents catastrophic forgetting of pretrained features
- Separate gradient clipping optional (backbone grads typically 5-10x larger)

**EMA (Exponential Moving Average):**
- Enabled by default with 0.9999 decay
- Automatically saved with checkpoints
- Used for evaluation when `ema.use_for_eval: true`
- Loading with `--load` prefers EMA weights unless `--no-ema` specified

**Checkpoint loading:**
- `--load`: Model weights only, fresh optimizer/scheduler
- `--resume-from-checkpoint`: Full training state (model, optimizer, scheduler, step)
- Automatic EMA weight detection and loading
- Shape mismatch filtering for robust loading

**Augmentation order:**
1. Load raw image
2. Clean invalid bboxes
3. Apply mosaic (probabilistic, uses cached samples)
4. Update cache
5. Apply per-image transforms (resize, flips, rotation, photometric)
6. Output to model

## Documentation

- **RESEARCH.md:** Architecture decisions and training rationale
- **CLAUDE.md:** Project guidelines for Claude Code
- **AGENTS.md:** Agent definitions for complex tasks

## References

- [D-FINE: Redefine Regression Task in DETRs](https://arxiv.org/abs/2410.13842)
- [ICDAR 2023 Competition on Hierarchical Text Detection](https://arxiv.org/abs/2305.14962)
- [DocLayout-YOLO](https://arxiv.org/html/2410.12628v1)
- [DINOv3: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)
