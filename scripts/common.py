"""Shared utilities for PROTAC-Splitter scripts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_dataset_or_csv(
    input_csv: Optional[str],
    hub_dataset_id: str,
    hub_config: str = "clustered",
    hub_split: str = "test",
    hub_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Load a dataset from a local CSV file or from the HuggingFace Hub."""
    from datasets import load_dataset, Dataset
    if input_csv is not None:
        return Dataset.from_pandas(pd.read_csv(input_csv))
    return load_dataset(
        hub_dataset_id,
        hub_config,
        token=hub_token or os.getenv("HF_TOKEN"),
        cache_dir=cache_dir,
    )[hub_split]


def get_hub_token(provided: Optional[str] = None) -> str:
    """Resolve a HuggingFace token from the argument, .env, or HF_TOKEN env var.

    Raises:
        ValueError: If no token is found.
    """
    token = provided or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token not found. "
            "Set HF_TOKEN in a .env file or pass --hub-token."
        )
    return token


def ensure_output_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
