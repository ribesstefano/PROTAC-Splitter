"""Push a trained ChemProp bond classifier to the HuggingFace Hub.

Converts the Lightning checkpoint written by the ChemProp training pipeline into
the pickle-free ``config.json`` + ``model.pt`` format loaded by
``ChemPropSplitter.from_pretrained``, and uploads it together with a generated model
card (README.md) and a manifest.json as a single commit to a public HuggingFace Hub
model repo.

Usage:
    python scripts/push_chemprop_to_hf.py --checkpoint outputs/chemprop/best.ckpt
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tyro

from protac_splitter.chemprop import ChemPropSplitter
from scripts.common import get_hub_token

_MODEL_CARD_TEMPLATE = """---
license: mit
library_name: chemprop
tags:
  - chemistry
  - cheminformatics
  - protac
  - drug-discovery
  - graph-neural-network
  - bond-classification
---

# PROTAC-Splitter ChemProp

ChemProp bond classifier used by [PROTAC-Splitter](https://github.com/ribesstefano/PROTAC-Splitter) to split
PROTAC molecules (heterobifunctional degraders) into their E3 ligand, linker, and POI warhead substructures.

Given the intact PROTAC, a directed message-passing neural network (initialised from the
[CheMeleon](https://github.com/JacksonBurns/chemeleon) encoder) classifies every bond as
0 = other, 1 = the POI-side (`[*:1]`) attachment bond, or 2 = the E3-side (`[*:2]`) attachment bond.
A constrained decoder then picks exactly one bond of each class such that cutting them yields
three connected parts.

## Usage

```python
from protac_splitter import split_protac

result = split_protac("<PROTAC SMILES>", model="chemprop")
```

`split_protac()` auto-downloads and caches the model (requires `pip install 'protac-splitter[chemprop]'`).
It can also be loaded directly:

```python
from protac_splitter.chemprop import ChemPropSplitter

model = ChemPropSplitter.from_pretrained("{repo_id}")
model.split(["<PROTAC SMILES>"])  # -> ["e3.linker.poi"]
```

## Files

- `config.json` — architecture: message-passing and bond-head hyperparameters, training class weights.
- `model.pt` — PyTorch tensor `state_dict` (load with `torch.load(..., weights_only=True)`).
- `manifest.json` — model metadata: training run parameters, test metrics, and library versions.

## Data

Trained on the curated PROTAC-Splitter dataset: https://doi.org/10.5281/zenodo.15797309
"""


@dataclasses.dataclass
class Args:
    """Push a trained ChemProp bond classifier to the HuggingFace Hub."""

    checkpoint: str = "./outputs/chemprop/best.ckpt"
    """Path to the Lightning checkpoint (best.ckpt) from the ChemProp training pipeline.
    run.json and test_metrics.json next to it, if present, are added to the manifest."""

    repo_id: str = "ailab-bio/PROTAC-Splitter-ChemProp"
    """HuggingFace Hub model repo to push to."""

    hub_token: str = ""
    """HuggingFace token (defaults to HF_TOKEN in .env)."""

    private: bool = False
    """Push as a private repo instead of public."""

    commit_message: str = "Upload PROTAC-Splitter ChemProp model"
    """Commit message for the upload."""


def _package_version() -> str:
    try:
        return version("protac-splitter")
    except PackageNotFoundError:
        return "unknown"


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _build_manifest(checkpoint: Path, splitter: ChemPropSplitter) -> dict:
    return {
        "source_checkpoint": checkpoint.name,
        "pushed_at": datetime.now(timezone.utc).isoformat(),
        "protac_splitter_version": _package_version(),
        "library": "chemprop",
        "wrapper_class": "protac_splitter.chemprop.ChemPropSplitter",
        "config": splitter.config,
        "training_run": _read_json(checkpoint.parent / "run.json"),
        "test_metrics": _read_json(checkpoint.parent / "test_metrics.json"),
    }


def main(args: Args) -> None:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Converting {checkpoint} to ChemPropSplitter format...")
    token = get_hub_token(args.hub_token)
    splitter = ChemPropSplitter.from_checkpoint(checkpoint, device="cpu")
    manifest = _build_manifest(checkpoint, splitter)

    print(f"Uploading {checkpoint} (converted), manifest.json, and README.md to {args.repo_id}...")
    commit_url = splitter.push_to_hub(
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
        extra_files={
            "manifest.json": json.dumps(manifest, indent=2, default=str),
            "README.md": _MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id),
        },
    )
    print(f"Done: {commit_url}")


if __name__ == "__main__":
    main(tyro.cli(Args))
