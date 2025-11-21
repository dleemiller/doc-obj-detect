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
    dry_run: bool


def cleanup_outputs(root: str | Path, *, dry_run: bool = False) -> CleanupSummary:
    """Remove output directories that never produced a checkpoint."""

    root_path = Path(root)
    if not root_path.exists():
        return CleanupSummary(root=root_path, candidates=[], removed=[], dry_run=dry_run)

    candidates = [path for path in root_path.iterdir() if _is_incomplete_run(path)]
    removed: list[Path] = []

    if not dry_run:
        for candidate in candidates:
            shutil.rmtree(candidate, ignore_errors=True)
            removed.append(candidate)

    return CleanupSummary(root=root_path, candidates=candidates, removed=removed, dry_run=dry_run)


def _is_incomplete_run(path: Path) -> bool:
    if not path.is_dir():
        return False

    final_model = path / "final_model"
    if final_model.exists():
        return False

    checkpoint_dir = path / "checkpoints"
    if not checkpoint_dir.is_dir():
        return False

    for item in checkpoint_dir.iterdir():
        if _looks_like_checkpoint(item):
            return False

    return True


def _looks_like_checkpoint(path: Path) -> bool:
    if "checkpoint" in path.name.lower():
        return True
    if path.suffix.lower() in {".bin", ".pt", ".pth"}:
        return True
    return False


__all__ = ["cleanup_outputs", "CleanupSummary"]
