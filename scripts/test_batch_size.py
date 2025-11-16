"""Test script to determine optimal batch size for D-FINE training."""

import gc

import torch

from doc_obj_detect.config import load_train_config
from doc_obj_detect.model import create_model


def test_batch_size(
    config_path: str = "configs/pretrain_publaynet.yaml",
    start_batch: int = 8,
    max_batch: int = 128,
    step: int = 8,
):
    """Test different batch sizes to find maximum that fits in VRAM.

    Args:
        config_path: Path to training configuration
        start_batch: Starting batch size to test
        max_batch: Maximum batch size to try
        step: Increment step for batch size
    """
    print("=" * 80)
    print("D-FINE Batch Size Testing")
    print("=" * 80)

    # Load config
    config = load_train_config(config_path)

    print("\nConfiguration:")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Image size: {config.data.image_size}")
    print(f"  Mixed precision: BF16={config.training.bf16}")

    # Create model
    print("\nInitializing model...")
    model, image_processor = create_model(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        use_pretrained_backbone=config.model.use_pretrained_backbone,
        freeze_backbone=config.model.freeze_backbone,
        image_size=config.data.image_size,
        **config.dfine.model_dump(),
    )

    # Move to GPU and set to training mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("⚠️  WARNING: No CUDA device found, testing on CPU (not realistic)")
    else:
        print(f"✅ Using device: {device}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Total VRAM: {total_memory:.1f} GB")

    model = model.to(device)
    model.train()

    # Enable mixed precision if configured
    use_amp = config.training.bf16
    if use_amp and device.type == "cuda":
        print("✅ Using BF16 mixed precision")
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("\n" + "=" * 80)
    print("Testing batch sizes...")
    print("=" * 80)

    results = []
    image_size = config.data.image_size

    batch_size = start_batch
    while batch_size <= max_batch:
        # Clear cache before each test
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        try:
            print(f"\n{'─' * 80}")
            print(f"Testing batch_size={batch_size}")
            print(f"{'─' * 80}")

            # Create dummy inputs
            pixel_values = torch.randn(
                batch_size, 3, image_size, image_size, device=device, dtype=dtype
            )
            pixel_mask = torch.ones(batch_size, image_size, image_size, device=device)

            # Create dummy labels (2 objects per image)
            labels = []
            for _ in range(batch_size):
                labels.append(
                    {
                        "class_labels": torch.tensor([0, 1], device=device),
                        "boxes": torch.tensor(
                            [[0.5, 0.5, 0.3, 0.3], [0.2, 0.2, 0.1, 0.1]],
                            device=device,
                            dtype=dtype,
                        ),
                    }
                )

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # Forward pass
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss

            # Backward pass
            loss.backward()

            # Clear gradients
            model.zero_grad()

            # Check memory usage
            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                current_memory = torch.cuda.memory_allocated() / 1024**3
                print("  ✅ SUCCESS")
                print(f"     Peak VRAM: {peak_memory:.2f} GB")
                print(f"     Current VRAM: {current_memory:.2f} GB")
                print(f"     Loss: {loss.item():.4f}")

                results.append(
                    {
                        "batch_size": batch_size,
                        "peak_vram_gb": peak_memory,
                        "current_vram_gb": current_memory,
                        "loss": loss.item(),
                        "success": True,
                    }
                )
            else:
                print("  ✅ SUCCESS (CPU)")
                print(f"     Loss: {loss.item():.4f}")
                results.append({"batch_size": batch_size, "loss": loss.item(), "success": True})

            # Increment batch size
            batch_size += step

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  ❌ OOM ERROR")
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"     VRAM at failure: {allocated:.2f} GB")

                results.append({"batch_size": batch_size, "success": False, "error": "OOM"})

                # OOM means we've found the limit
                break
            else:
                # Different error, re-raise
                raise

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not results:
        print("❌ No successful runs")
        return

    successful = [r for r in results if r["success"]]
    if not successful:
        print("❌ No successful runs")
        return

    max_batch = successful[-1]

    print(f"\n✅ Maximum working batch size: {max_batch['batch_size']}")
    if "peak_vram_gb" in max_batch:
        print(f"   Peak VRAM usage: {max_batch['peak_vram_gb']:.2f} GB")
        print(f"   VRAM headroom: {total_memory - max_batch['peak_vram_gb']:.2f} GB")

    print("\nAll successful runs:")
    print(f"{'Batch Size':<12} {'Peak VRAM (GB)':<15} {'Loss':<10}")
    print("─" * 40)
    for r in successful:
        if "peak_vram_gb" in r:
            print(f"{r['batch_size']:<12} {r['peak_vram_gb']:<15.2f} {r['loss']:<10.4f}")
        else:
            print(f"{r['batch_size']:<12} {'N/A (CPU)':<15} {r['loss']:<10.4f}")

    # Recommendations
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print("=" * 80)

    max_working_batch = max_batch["batch_size"]

    # Recommend 80-90% of max for stability
    recommended = int(max_working_batch * 0.85)
    # Round down to nearest multiple of 8
    recommended = (recommended // 8) * 8

    print(f"\n✅ Recommended batch size: {recommended}")
    print("   (85% of max for gradient accumulation and stability)")

    if recommended >= 64:
        print("\n✅ Config value of 64 should work well!")
    elif recommended >= 32:
        print("\n⚠️  Config value of 64 may be too high")
        print(f"   Recommend updating to batch_size={recommended}")
    else:
        print("\n❌ Config value of 64 is too high")
        print(f"   Recommend updating to batch_size={recommended}")
        print("   Or use gradient_accumulation_steps to increase effective batch size")

    # Gradient accumulation recommendation
    if recommended < 64:
        grad_accum = 64 // recommended
        print("\nTo achieve effective batch size of 64:")
        print(f"  Set batch_size={recommended}")
        print(f"  Set gradient_accumulation_steps={grad_accum}")
        print(f"  Effective batch size: {recommended * grad_accum}")


if __name__ == "__main__":
    test_batch_size()
