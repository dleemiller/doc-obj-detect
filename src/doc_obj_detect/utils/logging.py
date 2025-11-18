"""Centralized logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(run_name: str, log_dir: str | Path, level: int = logging.INFO) -> Path:
    """Configure a simple file+stdout logger for a training run.

    Args:
        run_name: Name of the current run (used in log headers).
        log_dir: Directory where log files should live.
        level: Logging level (defaults to INFO).

    Returns:
        Path to the log file created for this run.
    """

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logfile = log_path / f"{run_name}.log"

    # Reset root handlers so repeated CLI invocations don't accumulate handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logfile, mode="w"),
        ],
    )

    logging.info("Logging initialized for run %s", run_name)
    logging.info("Logs directory: %s", log_path)
    return logfile
