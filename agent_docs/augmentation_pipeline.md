# Augmentation Pipeline Architecture

**Location**: `src/doc_obj_detect/data/augmentor.py`

## Pipeline Structure

The augmentation system uses a **three-stage sequential pipeline**:

1. **Geometric** (always applied)
2. **Photometric OR Augraphy** (mutually exclusive, controlled by `choice_probability`)
3. **Mosaic** (batch-level, optional)

## Stage 1: Geometric Transforms

**Function**: `AlbumentationsAugmentor.build_geometric_transform()` (line 462)

**Always applied before photometric/Augraphy**. Affects spatial layout only:

- **Perspective**: `src/doc_obj_detect/data/augmentor.py:473-483`
- **Horizontal flip**: Line 486-488
- **Vertical flip**: Line 490-492
- **Rotation**: Line 495-503 (uses `rotate_limit` from config, typically 1-5°)
- **Random crop**: Applied separately based on config

**Bbox handling**: COCO format, min_visibility=0.3

## Stage 2A: Photometric Transforms (Classic CV)

**Function**: `AlbumentationsAugmentor.build_photometric_transform()` (line 514)

**Applied when Augraphy is NOT chosen**. All transforms have individual probabilities:

- **Brightness/contrast**: p=0.5, adjusts ±limit (line 525-533)
- **Blur**: p=0.3, OneOf[MotionBlur, GaussianBlur] (line 536-546)
- **JPEG compression**: p=0.3, quality 75-100 (line 549-558)
- **Gaussian noise**: p=0.3, std 0.0-0.01 (line 561-570)
- **Elastic transform**: Configurable p, distortion effect (line 573-581)
- **Sobel edge**: Optional, disabled by default (line 584-587)

## Stage 2B: Augraphy Pipeline (Document Degradation)

**Function**: `AlbumentationsAugmentor._build_augraphy_pipeline()` (line 104)

**Applied when choice_probability rolls Augraphy**. Three phases:

### Ink Phase (line 157-217)
- InkColorSwap, LinesDegradation, OneOf[Dithering, InkBleed]
- InkMottling, OneOf[LowInkRandomLines, LowInkPeriodicLines]
- Controlled by `ink_probability` config parameter

### Paper Phase (line 219-313)
- ColorPaper, OneOf[PatternGenerator, SubtleNoise, DirtyRollers, DoubleExposure]
- WaterMark, BrightnessTexturize, various texture effects
- Controlled by `paper_probability` config parameter

### Post Phase (line 315-371)
- OneOf[Brightness, Gamma, Faxify] - prevents stacking
- Final quality adjustments
- Controlled by `post_probability` config parameter

**Key setting**: `choice_probability` in config controls split:
- 0.0 = always photometric
- 1.0 = always Augraphy
- 0.5 = 50/50 mix (default)

## Stage 3: Mosaic (Batch-Level)

**Function**: `AlbumentationsAugmentor.build_mosaic_transform()` (line 598)

**Batch-based augmentation** (not per-image). Combines 4 images into grid.

- Requires sample cache (line 85-87, cache_size=20)
- Can be disabled after N epochs via `disable_after_epoch` config
- Probability controlled by `mosaic.probability` config parameter

## Configuration

See examples:
- `configs/pretrain_publaynet.yaml` - Production pretraining config
- `configs/finetune_doclaynet.yaml` - Fine-tuning config
- `configs/augmentation_example_with_augraphy.yaml` - Augraphy showcase

## Visualization

Generate comparison triplets (original, photometric, augraphy):
```bash
uv run doc-obj-detect visualize \
    --dataset publaynet \
    --mode comparison \
    --config configs/pretrain_publaynet.yaml \
    --num-samples 10 \
    --output-dir outputs/viz
```

Creates: `sample_XXX_1_original.jpg`, `sample_XXX_2_photometric.jpg`, `sample_XXX_3_augraphy.jpg`
