"""Runtime configuration: reads .env then environment variables."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Native math libraries (OpenMP, OpenBLAS, MKL, Apple Accelerate) each read their own
# thread-count env var and default to "use every core the process can see" — which on
# a cgroup-limited container can wildly exceed the actual CPU quota and turn a
# millisecond-scale single-row prediction into a multi-minute stall. setdefault
# respects any existing user override.
for _threads_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")


def get_cache_dir() -> Path:
    """Return the model cache directory (from PROTAC_SPLITTER_CACHE_DIR or default)."""
    return Path(os.getenv("PROTAC_SPLITTER_CACHE_DIR", "~/.cache/protac_splitter")).expanduser()


def get_hf_token() -> str | None:
    """Return the HuggingFace token from .env / environment, or None."""
    return os.getenv("HF_TOKEN")
