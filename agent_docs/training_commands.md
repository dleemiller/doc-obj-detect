# Training Commands Reference

**CLI Entry Point**: `src/doc_obj_detect/cli/main.py`

All commands use: `uv run doc-obj-detect <subcommand> [options]`

## Core Training Commands

### train
**Handler**: `_handle_train()` (line 209)

Start or resume training:
```bash
# Start new training
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml

# Resume from checkpoint (full state: optimizer, scheduler, step)
uv run doc-obj-detect train \
    --config configs/pretrain_publaynet.yaml \
    --resume-from-checkpoint outputs/pretrain_publaynet/checkpoint-5000

# Load weights only (fresh optimizer/scheduler)
uv run doc-obj-detect train \
    --config configs/finetune_doclaynet.yaml \
    --load outputs/pretrain_publaynet/checkpoint-best

# Load regular weights instead of EMA
uv run doc-obj-detect train \
    --config configs/finetune_doclaynet.yaml \
    --load outputs/pretrain_publaynet/checkpoint-best \
    --no-ema
```

**Key distinction**: `--resume` continues training, `--load` starts fresh with pretrained weights.

### evaluate
**Handler**: `_handle_evaluate()` (line 225)

Run evaluation on checkpoint:
```bash
uv run doc-obj-detect evaluate \
    --checkpoint outputs/pretrain_publaynet/checkpoint-best \
    --config configs/pretrain_publaynet.yaml \
    --batch-size 64 \
    --num-workers 0 \
    --max-samples 1000
```

**Note**: `--num-workers 0` avoids multiprocessing issues. Increase for production.

### distill
**Handler**: `_handle_distill()` (line 235)

Knowledge distillation (not yet implemented):
```bash
uv run doc-obj-detect distill --config configs/distill.yaml
```

## Utility Commands

### visualize
**Handler**: `_handle_visualize()` (line 254)

Preview augmentations:
```bash
# Simple mode: original vs augmented
uv run doc-obj-detect visualize \
    --dataset publaynet \
    --num-samples 4

# Comparison mode: original vs photometric vs augraphy
uv run doc-obj-detect visualize \
    --dataset publaynet \
    --mode comparison \
    --config configs/pretrain_publaynet.yaml \
    --num-samples 20 \
    --output-dir outputs/viz_comparison
```

### finalize
**Handler**: `_handle_finalize()` (line 265)

Prepare checkpoint for HuggingFace upload:
```bash
uv run doc-obj-detect finalize \
    outputs/pretrain_publaynet/checkpoint-best \
    models/publaynet-dfine-large

# Use regular weights instead of EMA
uv run doc-obj-detect finalize \
    outputs/pretrain_publaynet/checkpoint-best \
    models/publaynet-dfine-large \
    --no-ema

# Skip README generation
uv run doc-obj-detect finalize \
    outputs/pretrain_publaynet/checkpoint-best \
    models/publaynet-dfine-large \
    --no-readme
```

Reduces checkpoint size ~75% (removes optimizer, scheduler, training artifacts).

### dataset-info
**Handler**: `_handle_dataset_info()` (line 260)

Quick dataset summary:
```bash
uv run doc-obj-detect dataset-info --dataset publaynet
uv run doc-obj-detect dataset-info --dataset doclaynet --cache-dir /path/to/cache
```

### bbox-hist
**Handler**: `_handle_bbox_hist()` (line 277)

Analyze bounding box size distributions:
```bash
# Collect and plot from dataset
uv run doc-obj-detect bbox-hist \
    --config configs/pretrain_publaynet.yaml \
    --split val \
    --max-samples 1000 \
    --output outputs/bbox_hist.png \
    --data-out bbox_stats.npz

# Load precomputed stats
uv run doc-obj-detect bbox-hist \
    --data-in bbox_stats.npz \
    --output outputs/bbox_hist.png
```

### analyze-multiscale
**Handler**: `_handle_analyze_multiscale()` (line 342)

Analyze bbox statistics across multiple scaling policies:
```bash
uv run doc-obj-detect analyze-multiscale \
    --config configs/pretrain_publaynet.yaml \
    --split validation \
    --max-samples 5000 \
    --output-dir outputs/multiscale_analysis
```

### cleanup
**Handler**: `_handle_cleanup()` (line 368)

Remove incomplete training runs:
```bash
# Dry run (show what would be removed)
uv run doc-obj-detect cleanup --dry-run

# Actually remove
uv run doc-obj-detect cleanup

# Multiple roots
uv run doc-obj-detect cleanup --root outputs --root experiments

# Adjust TensorBoard log threshold
uv run doc-obj-detect cleanup --min-tb-log-size 100
```

Removes directories without checkpoints or with minimal TensorBoard logs.

## Configuration Files

**Location**: `configs/`

Key configs:
- `pretrain_publaynet.yaml` - PubLayNet pretraining (production)
- `finetune_doclaynet.yaml` - DocLayNet fine-tuning
- `architectures/dfine_xlarge.yaml` - D-FINE architecture params
- `augmentation_example_with_augraphy.yaml` - Augmentation showcase

## Verification

After training changes, verify with:
```bash
# Run tests
uv run pytest

# Check GPU memory
nvidia-smi

# Validate checkpoint loading
uv run doc-obj-detect evaluate --checkpoint outputs/.../checkpoint-X --config configs/...yaml
```
