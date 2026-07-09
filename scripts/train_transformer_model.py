"""Train the seq2seq Transformer model for PROTAC splitting.

Wraps ``protac_splitter.llms.training.train_model`` with a tyro CLI.
All arguments are inferred from the function's type-annotated signature.

Usage:
    python scripts/train_transformer_model.py --help
    python scripts/train_transformer_model.py --model-id MyModel --ds-name data/finetune
"""
from __future__ import annotations

import tyro
from protac_splitter.llms.training import train_model

if __name__ == "__main__":
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    tyro.cli(train_model)
