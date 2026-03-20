"""汎用ヘルパー関数"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import settings


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dirs():
    for d in (settings.CACHE_DIR, settings.MODELS_DIR, settings.RESULTS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
