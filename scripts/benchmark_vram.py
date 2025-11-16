#!/usr/bin/env python3
"""VRAM benchmarking script to determine optimal training configurations.

Tests different combinations of:
- Backbone models (base, large, gigantic)
- Batch sizes
- Gradient accumulation steps

Records actual VRAM usage, training speed, and OOM limits.
"""

import subprocess
import sys
import time
from pathlib import Path

import yaml

# Test configurations (backbone, name, image_size)
BACKBONES = [
    ("timm/vit_pe_spatial_base_patch16_512.fb", "base", 512),
    ("timm/vit_pe_spatial_large_patch14_448.fb", "large", 448),
    ("timm/vit_pe_spatial_gigantic_patch14_448.fb", "gigantic", 448),
]

# Test with small number of steps
TEST_STEPS = 10
WARMUP_STEPS = 2


def get_gpu_memory_nvidia_smi():
    """Get current GPU memory usage using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        memory_mb = float(result.stdout.strip())
        return memory_mb / 1024  # Convert to GB
    except Exception:
        return 0.0


def create_test_config(backbone: str, batch_size: int, grad_accum: int, image_size: int) -> dict:
    """Create test configuration dict."""
    return {
        "model": {
            "backbone": backbone,
            "num_classes": 5,
            "use_pretrained_backbone": True,
            "freeze_backbone": False,
        },
        "detr": {
            "num_queries": 300,
            "encoder_layers": 6,
            "decoder_layers": 6,
            "encoder_attention_heads": 8,
            "decoder_attention_heads": 8,
            "encoder_ffn_dim": 2048,
            "decoder_ffn_dim": 2048,
            "dropout": 0.1,
            "activation_function": "relu",
            "auxiliary_loss": True,
        },
        "data": {
            "dataset": "publaynet",
            "train_split": "train",
            "val_split": "validation",
            "image_size": image_size,
            "batch_size": batch_size,
            "num_workers": 4,
            "cache_dir": None,
        },
        "training": {
            "num_train_epochs": 1,
            "max_steps": TEST_STEPS,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_steps": WARMUP_STEPS,
            "gradient_accumulation_steps": grad_accum,
            "eval_strategy": "no",
            "eval_steps": 500,  # Won't be reached since max_steps=10
            "save_strategy": "no",
            "save_steps": 500,  # Won't be reached since max_steps=10
            "logging_steps": 5,
            "bf16": True,
            "dataloader_prefetch_factor": 2,
            "dataloader_persistent_workers": True,
        },
        "output": {
            "output_dir": "outputs/benchmark",
            "log_dir": "outputs/benchmark/logs",
        },
    }


def run_benchmark_test(
    backbone: str, backbone_name: str, batch_size: int, grad_accum: int, image_size: int
) -> dict | None:
    """Run a single benchmark test and return results."""
    print(
        f"\n{'='*80}\nTesting: {backbone_name} | batch_size={batch_size} | grad_accum={grad_accum}"
    )
    print(f"Effective batch size: {batch_size * grad_accum}")
    print("=" * 80)

    # Create config
    config = create_test_config(backbone, batch_size, grad_accum, image_size)
    config_path = Path("/tmp/benchmark_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Clear GPU memory before test
    subprocess.run(["nvidia-smi", "--gpu-reset"], capture_output=True)
    time.sleep(2)

    baseline_memory = get_gpu_memory_nvidia_smi()

    # Run training and monitor memory in parallel
    start_time = time.time()
    try:
        # Start training process
        proc = subprocess.Popen(
            ["uv", "run", "train", "--config", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Monitor memory usage while training runs
        peak_memory = baseline_memory
        memory_samples = []

        while proc.poll() is None:
            current_memory = get_gpu_memory_nvidia_smi()
            memory_samples.append(current_memory)
            if current_memory > peak_memory:
                peak_memory = current_memory
            time.sleep(0.5)

        # Wait for process to complete
        stdout, stderr = proc.communicate(timeout=60)
        elapsed_time = time.time() - start_time

        # Check if OOM occurred
        if proc.returncode != 0:
            if "out of memory" in stderr.lower() or "cuda error" in stderr.lower():
                print(f"❌ OOM Error - batch_size={batch_size} too large")
                return None
            else:
                print(f"❌ Training failed with error code {proc.returncode}")
                print(f"Error: {stderr[:500]}")
                return None

        # Extract training speed from output
        lines = stdout.split("\n")
        speed = None
        for line in lines:
            if "it/s" in line:
                try:
                    # Look for pattern like "4.12it/s"
                    parts = line.split("it/s")
                    if parts:
                        speed_str = parts[0].split()[-1]
                        speed = float(speed_str)
                        break
                except (ValueError, IndexError):
                    pass

        print("✅ Success!")
        print(f"   Peak VRAM: {peak_memory:.2f} GB")
        print(
            f"   Average VRAM: {sum(memory_samples)/len(memory_samples) if memory_samples else 0:.2f} GB"
        )
        print(f"   Training speed: {speed:.3f} it/s" if speed else "   Training speed: N/A")
        print(f"   Time elapsed: {elapsed_time:.1f}s")

        return {
            "backbone": backbone_name,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "peak_memory_gb": peak_memory,
            "avg_memory_gb": sum(memory_samples) / len(memory_samples) if memory_samples else 0,
            "speed_it_per_s": speed,
            "time_seconds": elapsed_time,
            "success": True,
        }

    except subprocess.TimeoutExpired:
        proc.kill()
        print("⏱️  Timeout - test took too long")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        if proc and proc.poll() is None:
            proc.kill()
        return None


def find_max_batch_size(backbone: str, backbone_name: str, image_size: int) -> int:
    """Binary search to find maximum viable batch size."""
    print(f"\n{'#'*80}\nFinding maximum batch size for {backbone_name}\n{'#'*80}")

    # Start with conservative bounds
    low, high = 1, 32
    max_working = 1

    while low <= high:
        mid = (low + high) // 2
        print(f"\nTrying batch_size={mid} (low={low}, high={high})")
        result = run_benchmark_test(
            backbone, backbone_name, mid, grad_accum=1, image_size=image_size
        )

        if result is not None:
            max_working = mid
            low = mid + 1
        else:
            high = mid - 1

    print(f"\n{'='*80}")
    print(f"Maximum batch size for {backbone_name}: {max_working}")
    print(f"{'='*80}\n")
    return max_working


def main():
    """Run comprehensive VRAM benchmarks."""
    # Check nvidia-smi is available
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except Exception:
        print("ERROR: nvidia-smi not available. Cannot run benchmarks.")
        sys.exit(1)

    # Print GPU info
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"GPU: {result.stdout.strip()}")

    results = []

    # Test each backbone
    for backbone, backbone_name, image_size in BACKBONES:
        print(f"\n{'#'*80}")
        print(f"TESTING BACKBONE: {backbone_name} (image_size={image_size})")
        print(f"{'#'*80}\n")

        # Find maximum batch size
        max_batch = find_max_batch_size(backbone, backbone_name, image_size)

        # Test a few batch sizes around the maximum
        if max_batch > 4:
            test_batches = [max_batch // 4, max_batch // 2, max_batch]
        else:
            test_batches = [1, max_batch] if max_batch > 1 else [1]

        test_batches = sorted({b for b in test_batches if b > 0})

        for batch_size in test_batches:
            # Test with different gradient accumulation
            for grad_accum in [1, 2, 4]:
                result = run_benchmark_test(
                    backbone, backbone_name, batch_size, grad_accum, image_size
                )
                if result:
                    results.append(result)

    # Print summary
    print(f"\n{'#'*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'#'*80}\n")

    print(
        f"{'Backbone':<12} {'Batch':<7} {'GradAcc':<9} {'Effective':<10} {'VRAM(GB)':<11} {'Speed(it/s)':<12}"
    )
    print("-" * 80)

    for r in results:
        speed_str = f"{r['speed_it_per_s']:.3f}" if r.get("speed_it_per_s") else "N/A"
        print(
            f"{r['backbone']:<12} {r['batch_size']:<7} {r['grad_accum']:<9} "
            f"{r['effective_batch_size']:<10} {r['peak_memory_gb']:<11.2f} {speed_str:<12}"
        )

    # Save results
    output_file = Path("outputs/benchmark_results.yaml")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {output_file}")

    # Print recommendations
    print(f"\n{'#'*80}")
    print("RECOMMENDATIONS")
    print(f"{'#'*80}\n")

    for _, backbone_name, _ in BACKBONES:
        backbone_results = [r for r in results if r["backbone"] == backbone_name]
        if backbone_results:
            # Find best throughput (highest effective batch size that fits)
            max_result = max(backbone_results, key=lambda x: x["effective_batch_size"])
            print(f"{backbone_name}:")
            print(f"  Max effective batch size: {max_result['effective_batch_size']}")
            print(
                f"  Config: batch_size={max_result['batch_size']}, "
                f"grad_accum={max_result['grad_accum']}"
            )
            print(f"  VRAM usage: {max_result['peak_memory_gb']:.2f} GB")
            if max_result.get("speed_it_per_s"):
                print(f"  Training speed: {max_result['speed_it_per_s']:.3f} it/s")
            print()


if __name__ == "__main__":
    main()
