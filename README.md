# Document Layout Analysis with D-FINE

Modern document understanding needs both page-level accuracy and production-ready throughput. This project trains a ConvNeXt‑DINOv3 backbone paired with the D‑FINE detector, fine‑tunes on DocLayNet, and then distills into smaller D‑FINE variants (ConvNeXt Base / Small / Tiny). All tooling flows through `uv run doc-obj-detect …`, so dependencies stay reproducible and CUDA wheels are easy to swap.

## Highlights
- **Teacher architecture:** ConvNeXt-Large DINOv3 backbone feeding D-FINE (keeps VRAM low enough for a 3-level stride pyramid) with 300 queries
- **Aspect-preserving multiscale:** Short side sampled from `[576, 608, 640]`, long side capped at 928 px; evaluation always resizes to a fixed 640 px short side with no augmentation
- **Students:** Smaller ConvNeXt backbones (Base / Small / Tiny) paired with D-FINE variants; distillation loss supports KL (with temperature) or MSE on logits plus optional box-level MSE
- **Datasets:** PubLayNet pretraining (5 classes) followed by DocLayNet finetuning (11 classes). Config schemas enforce sane ranges for image sizes, batch sizes, learning rates, multiscale settings, etc., so misconfigured runs fail fast
- **Configs:** `configs/pretrain_publaynet.yaml`, `configs/finetune_doclaynet.yaml`, and `configs/distill.yaml` describe the entire pipeline; modify YAML rather than editing code
- **Architecture flexibility:** D-FINE variants (XLarge/Large/Medium/Small) defined in `configs/architectures/` for easy experimentation

## New Features (Branch: cleanup)

### ✅ Comprehensive Test Suite (69% coverage)
- **33 new tests** covering critical components:
  - EMA checkpoint loading (fixed bug with TDD approach)
  - Split LR optimizer and custom gradient clipping
  - Metrics computation (COCO-style mAP)
  - Coordinate transformations and edge cases
- All tests run on CPU to avoid GPU interference
- Run with: `uv run pytest --cov=src`

### ✅ Enhanced Checkpoint Loading
- **EMA weight loading:** Automatically loads EMA weights from training checkpoints when available
- **Flexible loading modes:**
  - Fresh start: Build model from config
  - `--resume-from-checkpoint`: Full state (model, optimizer, scheduler, step)
  - `--load`: Model weights only (start new training with pretrained weights)
  - `--no-ema`: Force loading regular weights instead of EMA
- **Robust handling:** Shape mismatch filtering, directory vs file path support

### ✅ Split Learning Rates & Gradient Clipping
- **Backbone LR multiplier:** `backbone_lr_multiplier: 0.01` (backbone gets 1% of head LR)
- **Separate gradient clipping:** `backbone_max_grad_norm` and `head_max_grad_norm` (critical for training stability, backbone grads are 5-10x larger)
- **Gradual unfreezing:** `freeze_backbone_epochs: 2` for multi-phase training

### ✅ Architecture System
- D-FINE variants in `configs/architectures/`:
  - `dfine_xlarge.yaml` - 6 decoder layers, silu activation (best accuracy)
  - `dfine_large.yaml` - 6 decoder layers, relu activation
  - `dfine_medium.yaml` - 4 decoder layers, depth_mult=0.67
  - `dfine_small.yaml` - 3 decoder layers, depth_mult=0.34, hidden_expansion=0.5
- Switch architectures by changing one line: `model.architecture: dfine_large`

### ✅ EMA Callback
- **Exponential Moving Average** for smoother weights during evaluation
- Automatically saves EMA state with checkpoints
- Configurable decay (default: 0.9999) and warmup steps (default: 1000)
- Use EMA weights for evaluation: `ema.use_for_eval: true`

## Happy Path: PubLayNet → DocLayNet → Distillation

### Phase 1: Pretrain on PubLayNet (frozen backbone)
```bash
# Start with frozen backbone for 2 epochs to stabilize head
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml

# Config highlights:
# - freeze_backbone: true
# - freeze_backbone_epochs: 2  # Auto-unfreezes after 2 epochs
# - backbone_lr_multiplier: 0.01  # 1% of head LR when unfrozen
# - backbone_max_grad_norm: 0.1  # Separate clipping for stability
# - head_max_grad_norm: 0.1
# - ema.enabled: true  # EMA for better eval metrics
```

**What happens:**
1. First 2 epochs: Backbone frozen, head trains with LR=1.25e-4
2. After epoch 2: Backbone unfreezes automatically
3. Epochs 3-12: End-to-end training with split LRs (head: 1.25e-4, backbone: 1.25e-6)
4. Separate gradient clipping prevents backbone grads from dominating
5. EMA weights saved throughout for stable evaluation

### Phase 2: Fine-tune on DocLayNet (load pretrained weights)
```bash
# Load best checkpoint from Phase 1 (EMA weights preferred)
uv run doc-obj-detect train \
  --config configs/finetune_doclaynet.yaml \
  --load outputs/pretrain_publaynet_dfine_phase2/checkpoint-5000

# Or load final EMA model:
uv run doc-obj-detect train \
  --config configs/finetune_doclaynet.yaml \
  --load outputs/pretrain_publaynet_dfine_phase2/final_model_ema

# To force non-EMA weights (debugging):
uv run doc-obj-detect train \
  --config configs/finetune_doclaynet.yaml \
  --load outputs/pretrain_publaynet_dfine_phase2/checkpoint-5000 \
  --no-ema

# Config highlights:
# - num_classes: 11  # DocLayNet has more classes than PubLayNet
# - freeze_backbone: false  # End-to-end training from start
# - backbone_lr_multiplier: 0.1  # Higher than pretraining (10% vs 1%)
# - learning_rate: 5.0e-5  # Lower base LR for fine-tuning
```

**What happens:**
1. Loads pretrained model weights (automatically uses EMA if available)
2. Reinitializes detection head for 11 classes (vs 5 in PubLayNet)
3. Fresh optimizer and scheduler (not resumed)
4. End-to-end training with higher backbone LR multiplier (0.1 vs 0.01)

### Phase 3: Distill to Smaller Variants
```bash
# Distill DocLayNet-finetuned model to smaller architecture
uv run doc-obj-detect distill --config configs/distill.yaml

# Modify configs/distill.yaml to experiment:
# - model.architecture: dfine_medium  # or dfine_small for faster inference
# - model.backbone: convnext_base.dinov2  # or convnext_small
# - distillation.alpha: 0.7  # Weight for distillation loss
# - distillation.beta: 0.3   # Weight for ground truth loss
```

**What happens:**
1. Teacher: Loads your fine-tuned DocLayNet model
2. Student: Smaller backbone + D-FINE variant (configurable architecture)
3. Trains with combined loss: `alpha * distill_loss + beta * gt_loss`
4. Produces deployment-ready model with lower latency

### Resume Interrupted Training
```bash
# Resume training from checkpoint (full state: model, optimizer, scheduler, step)
uv run doc-obj-detect train \
  --config configs/pretrain_publaynet.yaml \
  --resume-from-checkpoint outputs/pretrain_publaynet_dfine_phase2/checkpoint-3000

# Note: Cannot use both --resume and --load (mutually exclusive)
```

## Quick Start
```bash
git clone git@github.com:dleemiller/doc-obj-detect.git
cd doc-obj-detect

# Install dependencies
uv sync

# Run tests
uv run pytest --cov=src

# Start training
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml
```

### Training / Evaluation / Distillation (Unified CLI)
```bash
# Pretraining / finetuning
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml
uv run doc-obj-detect train --config configs/finetune_doclaynet.yaml

# Load weights from checkpoint
uv run doc-obj-detect train --config configs/finetune_doclaynet.yaml \
  --load outputs/pretrain_publaynet/checkpoint-5000

# Resume interrupted training
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml \
  --resume-from-checkpoint outputs/pretrain_publaynet/checkpoint-3000

# Checkpoint evaluation (short side fixed at 640 px, no augmentation)
uv run doc-obj-detect evaluate \
  --config configs/pretrain_publaynet.yaml \
  --checkpoint outputs/pretrain_publaynet_dfine/checkpoint-1234

# Distill teacher -> smaller D-FINE student
uv run doc-obj-detect distill --config configs/distill.yaml
```

### Monitoring & Utilities
- **TensorBoard:** `tensorboard --logdir outputs/<run>/logs`
- **Augmentation preview:** `uv run doc-obj-detect visualize --dataset publaynet`
- **Dataset info:** `uv run doc-obj-detect dataset-info --dataset doclaynet`
- **Coverage report:** `uv run pytest --cov=src --cov-report=html` (open `htmlcov/index.html`)

## Project Layout
```
src/doc_obj_detect/
├─ cli/                 # Central CLI entry point + subcommand handlers
├─ config/              # Pydantic schemas, YAML loaders, architecture merging
├─ data/                # HF dataset wrappers, Albumentations augmentor, collate_fn
├─ models/              # ModelFactory + checkpoint loading + parameter statistics
├─ training/
│   ├─ base_runner.py   # Shared dataset/processor/run-dir helpers
│   ├─ runner.py        # TrainerRunner (teacher training)
│   ├─ distill_runner.py# DistillRunner (student KD training)
│   ├─ distillation.py  # DistillationTrainer (KL/MSE KD losses)
│   ├─ evaluator.py     # EvaluatorRunner (checkpoint eval loop)
│   ├─ trainer_core.py  # SplitLRTrainer (custom HF Trainer with split LR/grad clipping)
│   └─ callbacks.py     # EMACallback, UnfreezeBackboneCallback
├─ metrics.py           # COCO-style mAP via torchmetrics
└─ visualize.py         # Augmentation visualization utility

configs/
├─ architectures/       # D-FINE architecture definitions (XLarge/Large/Medium/Small)
├─ pretrain_publaynet.yaml
├─ finetune_doclaynet.yaml
└─ distill.yaml

tests/                  # Pytest suites (69% coverage)
├─ test_model_loading.py    # Checkpoint loading, EMA weights
├─ test_trainer_core.py     # Split LR, gradient clipping
├─ test_metrics.py          # mAP computation, bbox transforms
└─ ...
```

## Key Settings

### Training Configuration
- **Optimizer:** AdamW fused, `learning_rate=1.25e-4`, gradient accumulation steps: 1
- **Batch size:** 16 images/device for long runs (multi-GPU data parallel supported)
- **Scheduler:** Cosine with `min_lr=1.25e-5` (precision tail for final mAP gains)
- **Mixed precision:** BF16 (better stability than FP16)
- **Gradient clipping:** Separate norms for backbone (0.1) and head (0.1)

### Model Configuration
- **Backbone:** ConvNeXt-Large DINOv3 (`convnext_large.dinov3_lvd1689m`)
- **Architecture:** D-FINE XLarge (6 decoder layers, encoder_hidden_dim=384)
- **Feature levels:** 3 (strides 8, 16, 32)
- **Queries:** 300
- **Backbone freeze:** First 2 epochs, then gradual unfreeze

### Evaluation
- **Image size:** Short edge fixed at 640 px with 928 px long-edge cap
- **No augmentations:** Clean evaluation
- **Metrics:** COCO-style mAP, mAR (via torchmetrics)
- **Max eval samples:** 1000-2000 (configurable to avoid OOM)

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Run specific test file
uv run pytest tests/test_model_loading.py -v

# Run specific test
uv run pytest tests/test_trainer_core.py::TestSplitLROptimizer::test_applies_lr_multiplier_correctly -v
```

**Test Coverage:**
- Overall: 69%
- Critical components: 88-99% (models/builder.py, metrics.py, training/callbacks.py)
- All tests run on CPU to avoid GPU interference

## Configuration Guide

### Architecture Selection
```yaml
model:
  architecture: dfine_xlarge  # Change to: dfine_large, dfine_medium, dfine_small
```

Architecture files in `configs/architectures/` define:
- Decoder layers (6/6/4/3 for XL/L/M/S)
- Hidden dimensions and scaling factors
- Loss weights and matcher costs
- Training hyperparameters

### Split Learning Rates
```yaml
training:
  learning_rate: 1.25e-4           # Head LR
  backbone_lr_multiplier: 0.01     # Backbone LR = 1% of head LR
```

### Gradient Clipping
```yaml
training:
  backbone_max_grad_norm: 0.1  # Critical: backbone grads are 5-10x larger
  head_max_grad_norm: 0.1
```

### Backbone Freezing
```yaml
model:
  freeze_backbone: true
  freeze_backbone_epochs: 2  # Unfreeze after 2 epochs
```

### EMA Configuration
```yaml
training:
  ema:
    enabled: true
    decay: 0.9999           # Smoothing factor
    warmup_steps: 1000      # Gradual EMA buildup
    use_for_eval: true      # Use EMA weights for evaluation
```

## Notes
- All scripts run through the CLI (`uv run doc-obj-detect ...`) so we don't accumulate separate entry points
- Distillation currently exposes logit KL/MSE loss with optional predicted-box MSE; GO-LSD-style distribution distillation is planned
- PubLayNet/DocLayNet downloads rely on Hugging Face Datasets; set `data.cache_dir` if `$HF_HOME` is not writable
- The `--load` flag automatically uses EMA weights when available; use `--no-ema` to force regular weights
- Test suite ensures correctness of split LR, gradient clipping, EMA loading, and metrics computation

## Documentation
- **RESEARCH.md:** Architecture decisions and training rationale (not an ablation log)
- **CLAUDE.md:** Project guidelines for Claude Code
- **AGENTS.md:** Agent definitions for complex tasks
- **Test coverage:** `htmlcov/index.html` after running `pytest --cov-report=html`

See [RESEARCH.md](./docs/RESEARCH.md) for design rationale and future work.
