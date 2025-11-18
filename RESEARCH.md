# Research Summary – ConvNeXt + D-FINE Pipeline

This document captures the design rationale behind the current document layout detection pipeline. It replaces the earlier ViT‑PE + Deformable DETR plan and describes how we now leverage ConvNeXt‑DINOv3 backbones with the D‑FINE detector, aspect-preserving multiscale training, and downstream distillation to real-time heads.

---

## Executive Overview

| Component | Choice | Motivation |
|-----------|--------|------------|
| Backbone | `convnext_large.dinov3_lvd1689m` (timm) | Strong ImageNet/Objects365 pretrained features, easier to fine-tune than 1.8 B ViT, fits comfortably on 96 GB VRAM |
| Detector Head | **D-FINE** (Fine-grained Distribution Refinement + GO-LSD) | Better localization than vanilla Deformable DETR, fast enough for near real-time |
| Image Pipeline | Aspect-preserving multiscale (short edge 480–640, long edge ≤ 928) | Matches the D-FINE paper’s recipe; keeps native document aspect ratios |
| Training Schedule | PubLayNet pretrain (12 epochs) → DocLayNet finetune (24 epochs) | Aligns with literature + our available compute |
| Students | RT-DETR-based heads fed by ConvNeXtV2/ViT-PE-Small | Distillation targets for deployment-friendly inference |

The core idea is: maximize teacher accuracy with D‑FINE + ConvNeXt on documents, then use GO-LSD/FDR knowledge to supervise smaller real-time detectors.

---

## 1. Architecture Selection

### 1.1 Backbone

- **ConvNeXt Large, DINOv3 pretraining**
  - Provides high-quality features with moderate parameter count (~198 M) versus the previous 1.8 B ViT.
  - Timm integrates smoothly with D-FINE via `use_timm_backbone=True`.
  - Empirically stable on document imagery (less overfitting than ViT-PE when data augmentation is limited).

- **Why not ViT-PE anymore?**
  - Training cost was too high for iterative experiments.
  - D-FINE’s encoder assumes multi-scale CNN stages; ConvNeXt is a better structural fit.
  - The SOTA D-FINE paper demonstrates strong results on ResNet/RT-DETR-style backbones, making it a safer baseline.

### 1.2 Detector Head – D-FINE

D-FINE introduces two mechanisms that are particularly relevant for dense document layouts:

1. **Fine-grained Distribution Refinement (FDR)**
   - Instead of regressing coordinates directly, each decoder layer outputs probability distributions for the four box edges.
   - Residual refinements across layers capture localization uncertainty and prevent jitter.
   - Weighted sums (Eq. 2 in the paper) let us adjust each edge with finer granularity, which is valuable for thin tables/paragraph regions.

2. **Global Optimal Localization Self-Distillation (GO-LSD)**
   - Takes the refined distributions from the final decoder layer and distills them into shallower decoder outputs.
   - Acts like built-in localization distillation, stabilizing training even with small batch sizes.
   - Minimal compute overhead compared to separate teacher/student setups.

Additional tweaks from the paper (Target Gating, uneven sampling points, GELAN encoders) mainly apply when designing lightweight variants. Our current use case keeps the default D-FINE config (deformable attention layers + FDR/GO-LSD losses) and relies on ConvNeXt for the multi-scale features.

---

## 2. Data Processing & Augmentation

### 2.1 Aspect-Preserving Multiscale

The D-FINE paper scales the short edge to various lengths during training and fixes it at 640 px for evaluation. We replicate this policy:

- **Training short edges:** `[480, 512, 544, 576, 608, 640]` selected per batch.
- **Long-edge cap:** 928 px (≈ 1.45×640) to cover the long tail of 1.41 aspect ratios observed in PubLayNet (see histogram shared earlier).
- **Padding:** After Albumentations, we pad to the batch’s max height/width, aligned to the stride of the detector (`max(config.dfine.feat_strides)`), so stacked tensors share a uniform shape.

This preserves each page’s native aspect while still providing the scale diversity the paper requires.

### 2.2 Augmentation Stack

Document-friendly Albumentations config (now expressed via nested config classes):

- Perspective + elastic transforms with modest probabilities (0.3 / 0.2).
- Small rotations (≤ 5°) and mild brightness/contrast jitter (0.2).
- Blur/compression/noise to mimic scanning artifacts.
- No random crops or color shifts that would destroy layout semantics.

---

## 3. Training Strategy

### 3.1 PubLayNet Pretraining

- **Epochs:** 12 (same as D-FINE COCO schedule).
- **Batch size:** 16 images per GPU (fits 96 GB with padding).
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=0.05`, gradient clipping at 1.0).
- **Scheduler:** Cosine (`min_lr=1e-5`).
- **Precision:** BF16 (matches DINOv3 finetuning conventions).

### 3.2 DocLayNet Finetuning

- **Epochs:** 24 (DocLayNet is smaller but has more classes; longer schedule helps).
- **Freeze policy:** Start with backbone frozen for a few thousand steps if gradients become unstable; otherwise train end-to-end.
- **Augmentation:** Same as PubLayNet to avoid domain shift.

### 3.3 Evaluation

- Evaluation processor forces `{"shortest_edge": 640, "longest_edge": 928}` so inference matches the paper’s “resize short side to 640” rule.
- COCO-style metrics via `torchmetrics` + HF post-processing.
- For DocLayNet we keep the default 11-class taxonomy; for PubLayNet we track the 5-class AP breakdown.

---

## 4. Distillation Plan

While D-FINE already includes GO-LSD (self-distillation), we still need deployment-friendly heads. The distillation module introduced in this refactor formalizes that pipeline:

1. **Teacher:** ConvNeXt-L + D-FINE (frozen) loaded via `DFineForObjectDetection.from_pretrained`.
2. **Students:** Smaller ConvNeXt variants (Base / Small / Tiny) driving the same D-FINE head, so we keep FDR + decoder semantics while shrinking the backbone:
   - **Base:** `encoder_in_channels=[192,384,768]`
   - **Small:** `[128,256,512]`
   - **Tiny:** `[96,192,384]`
   These mappings live in the new distillation config and model factory so we can change the backbone by editing YAML only.
3. **Distillation Losses (implemented):**
   - **Logit distillation:** configurable between KL (with temperature) and MSE.
   - **Feature distillation:** optional MSE on the predicted boxes (acts as a proxy for GO-LSD’s localization refinement).
   - **Loss mixing:** α (KD) + β (ground-truth) ensures we never lose anchor to real labels.
4. **Datasets + Augs:** Reuse the same `DatasetFactory` + Albumentations pipeline as the teacher so we do not introduce domain drift.

### 4.1 Future GO-LSD Feature Distillation (Not Yet Implemented)

To keep parity with the paper’s “Global Optimal Localization Self-Distillation” but across teacher→student, we will:

- **Capture teacher distributions:** Hook into the teacher’s final decoder layer to extract the refined probability distributions (`Pr_t`, `Pr_b`, etc.) that GO-LSD uses internally.
- **Student auxiliary heads:** Add a lightweight projection after each student decoder layer producing the same distribution format (bins × edges). These heads will only exist during training.
- **Distribution KD:** Minimize KL/MSE between teacher and student distributions for each matched query. Because these are already normalized (softmax), temperature is unnecessary; we can use standard KL or a smoothed L2.
- **Matching strategy:** Reuse the union-set matching from GO-LSD so we supervise both matched and unmatched queries, stabilizing dense layouts.
- **Scheduling:** Start with distribution KD disabled for a few epochs so the student learns coarse localization from the logit loss, then enable GO-LSD KD once the student loss plateaus.

Once this feature is implemented, the config will expose a `distill_go_lsd` block (bins, weighting, start epoch) but for now we are holding off until the base KD pipeline is fully validated. This plan ensures we can add GO-LSD-style feature transfer without reworking the current trainer.

---

## 5. Alternatives Considered

| Option | Why rejected |
|--------|--------------|
| ViT-PE + Deformable DETR | Too resource-intensive, doesn’t leverage D-FINE’s distribution head. |
| Faster/Cascade R-CNN | Anchor-heavy, less compatible with distribution-based regressions, worse small-object recall without extra tuning. |
| YOLOv11 | Great for speed but lacks the multi-query transformer pipeline needed to reuse FDR/GO-LSD knowledge; we still target YOLO-like latency via distilled RT-DETR. |

---

## 6. Expected Performance & Monitoring

- **PubLayNet target:** 95 ± 1 AP (mAP@0.5) with the teacher.
- **DocLayNet target:** ≥ 82 mAP (11-class).
- **Real-time students:** Aim for ≥ 90% teacher AP at ~10 ms latency (e.g., RT-DETR-L on T4 GPU per D-FINE’s Table 1).

Monitoring:

- Log loss/AP via TensorBoard; keep eval frequency modest (every 1–2k steps) to avoid excessive scale jumping mid-epoch.
- Watch gradient norms (multiscale causes spikes; we clamp at 1.0 but track histograms).

---

## References

1. **D-FINE:** Peng et al., *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*, arXiv:2410.13842.
2. **RT-DETR:** Zhao et al., *End-to-End Object Detection with Adaptive Clustering Transformers*.
3. **ConvNeXt / DINOv3:** Liu et al., *A ConvNet for the 2020s*; Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*.
4. **PubLayNet / DocLayNet** dataset papers.
5. **Albumentations** documentation for geometric/photometric transforms.
6. Prior research summary (archived) for historical context on ViT-PE experiments.
