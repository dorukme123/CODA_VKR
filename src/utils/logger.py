import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import LOGS_DIR


def setup_logger(
    name: str,
    log_dir: Path = LOGS_DIR,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}_{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # --- File handler ---
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # --- Console handler ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter(
        fmt="%(levelname)-8s | %(message)s",
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)

    return logger
