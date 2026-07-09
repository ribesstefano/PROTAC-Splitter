"""Runtime configuration: reads .env then environment variables."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# XGBoost with OpenMP can deadlock on macOS ARM64 when spawning multiple threads
# during model deserialization. setdefault respects any existing user override.
os.environ.setdefault("OMP_NUM_THREADS", "1")


def get_cache_dir() -> Path:
    """Return the model cache directory (from PROTAC_SPLITTER_CACHE_DIR or default)."""
    return Path(os.getenv("PROTAC_SPLITTER_CACHE_DIR", "~/.cache/protac_splitter")).expanduser()


def get_hf_token() -> str | None:
    """Return the HuggingFace token from .env / environment, or None."""
    return os.getenv("HF_TOKEN")
