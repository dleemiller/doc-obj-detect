"""Centralized logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(run_name: str, log_dir: str | Path, level: int = logging.INFO) -> Path:
    """Configure rich console + file logging for a training run."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logfile = log_path / f"{run_name}.log"

    # Reset handlers to avoid duplicate outputs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=False, markup=True),
            logging.FileHandler(logfile, mode="w"),
        ],
    )

    logging.getLogger("rich").setLevel(level)
    logging.info("Logging initialized for run %s", run_name)
    logging.info("Logs directory: %s", log_path)
    return logfile
