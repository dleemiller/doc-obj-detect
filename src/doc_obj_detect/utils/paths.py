"""Path helpers for training/evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from doc_obj_detect.config import OutputConfig


@dataclass
class RunPaths:
    output_dir: Path
    log_dir: Path
    run_name: str

    @property
    def final_model_dir(self) -> Path:
        return self.output_dir / "final_model"


def prepare_run_dirs(output_cfg: OutputConfig) -> RunPaths:
    """Ensure output/log directories exist and return their paths."""

    output_dir = Path(output_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always append timestamp to run name to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = output_cfg.run_name or "run"
    run_name = f"{base_name}_{timestamp}"

    if output_cfg.log_dir:
        # Create subdirectory with run_name under the specified log_dir
        log_dir = Path(output_cfg.log_dir) / run_name
    else:
        log_dir = output_dir / "logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(output_dir=output_dir, log_dir=log_dir, run_name=run_name)
