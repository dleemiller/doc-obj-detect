"""Helpers for pruning incomplete training runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CleanupSummary:
    root: Path
    candidates: list[Path]
    removed: list[Path]
    tb_log_candidates: list[Path]
    tb_logs_removed: list[Path]
    dry_run: bool


def cleanup_outputs(
    root: str | Path, *, dry_run: bool = False, min_tb_log_size_kb: int = 50
) -> CleanupSummary:
    """Remove output directories that never produced a checkpoint and minimal TensorBoard logs.

    Args:
        root: Root directory to scan for incomplete runs
        dry_run: If True, don't actually delete anything
        min_tb_log_size_kb: Minimum total size (KB) for TensorBoard logs to keep (default: 50KB)
    """

    root_path = Path(root)
    if not root_path.exists():
        return CleanupSummary(
            root=root_path,
            candidates=[],
            removed=[],
            tb_log_candidates=[],
            tb_logs_removed=[],
            dry_run=dry_run,
        )

    candidates = [path for path in root_path.iterdir() if _is_incomplete_run(path)]
    removed: list[Path] = []

    # Find minimal TensorBoard logs
    tb_log_candidates: list[Path] = []
    tb_logs_removed: list[Path] = []
    for path in root_path.iterdir():
        if path.is_dir():
            logs_dir = path / "logs"
            if logs_dir.exists() and logs_dir.is_dir():
                tb_log_candidates.extend(
                    _find_minimal_tb_logs(logs_dir, min_size_kb=min_tb_log_size_kb)
                )

    if not dry_run:
        for candidate in candidates:
            shutil.rmtree(candidate, ignore_errors=True)
            removed.append(candidate)

        for tb_log in tb_log_candidates:
            shutil.rmtree(tb_log, ignore_errors=True)
            tb_logs_removed.append(tb_log)

    return CleanupSummary(
        root=root_path,
        candidates=candidates,
        removed=removed,
        tb_log_candidates=tb_log_candidates,
        tb_logs_removed=tb_logs_removed,
        dry_run=dry_run,
    )


def _is_incomplete_run(path: Path) -> bool:
    """Check if a run directory is incomplete (no final model or checkpoints).

    A run is considered incomplete if:
    - No final_model directory exists
    - AND no checkpoint directories exist (either in checkpoints/ subdir or checkpoint-* pattern)
    """
    if not path.is_dir():
        return False

    # If final model exists, run is complete
    final_model = path / "final_model"
    if final_model.exists():
        return False

    # Check for checkpoint-* directories (HuggingFace Trainer pattern)
    has_checkpoints = False
    try:
        for item in path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                has_checkpoints = True
                break
    except (OSError, PermissionError):
        # If we can't read the directory, treat as incomplete
        return True

    if has_checkpoints:
        return False

    # Also check legacy checkpoints/ subdirectory pattern
    checkpoint_dir = path / "checkpoints"
    if checkpoint_dir.exists() and checkpoint_dir.is_dir():
        try:
            for item in checkpoint_dir.iterdir():
                if _looks_like_checkpoint(item):
                    return False
        except (OSError, PermissionError):
            pass

    # No checkpoints found, run is incomplete
    return True


def _looks_like_checkpoint(path: Path) -> bool:
    if "checkpoint" in path.name.lower():
        return True
    if path.suffix.lower() in {".bin", ".pt", ".pth"}:
        return True
    return False


def _find_minimal_tb_logs(logs_dir: Path, min_size_kb: int) -> list[Path]:
    """Find TensorBoard log directories with minimal data.

    Args:
        logs_dir: Directory containing TensorBoard run subdirectories
        min_size_kb: Minimum total size in KB to keep a log directory

    Returns:
        List of log directories that should be removed (have < min_size_kb data)
    """
    minimal_logs: list[Path] = []

    try:
        for run_dir in logs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Calculate total size of event files in this run
            total_size = 0
            event_count = 0
            try:
                for item in run_dir.iterdir():
                    if item.is_file() and "tfevents" in item.name:
                        total_size += item.stat().st_size
                        event_count += 1
            except (OSError, PermissionError):
                continue

            # Remove if total event data is less than threshold
            size_kb = total_size / 1024
            if event_count > 0 and size_kb < min_size_kb:
                minimal_logs.append(run_dir)

    except (OSError, PermissionError):
        pass

    return minimal_logs


__all__ = ["cleanup_outputs", "CleanupSummary"]
