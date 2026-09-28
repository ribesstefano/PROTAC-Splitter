"""Push a trained XGBoost graph edge classifier to the HuggingFace Hub.

Bundles the joblib model file together with a generated model card (README.md)
and a manifest.json (hyperparameters, feature lists, SHA-256 checksum) and
uploads them as a single commit to a public HuggingFace Hub model repo.

Usage:
    python scripts/push_xgboost_to_hf.py --model-path models/PROTAC-Splitter-XGBoost.joblib
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tyro
from huggingface_hub import HfApi

from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier
from scripts.common import get_hub_token

_MODEL_CARD_TEMPLATE = """---
license: mit
library_name: xgboost
tags:
  - chemistry
  - cheminformatics
  - protac
  - drug-discovery
  - graph-edge-classification
---

# PROTAC-Splitter XGBoost

XGBoost graph edge classifier used by [PROTAC-Splitter](https://github.com/ribesstefano/PROTAC-Splitter) to split
PROTAC molecules (heterobifunctional degraders) into their E3 ligand, linker, and POI warhead substructures.

Given a PROTAC's molecular graph, the model scores each candidate bond ("edge") for how likely it is to be
one of the two cleavage points that separate the three substructures.

## Usage

```python
from protac_splitter import split_protac

result = split_protac("<PROTAC SMILES>", model="xgboost")
```

`split_protac()` auto-downloads and caches the model. It can also be loaded directly:

```python
from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier

model = GraphEdgeClassifier.from_pretrained("{repo_id}")
```

## Files

- `{model_filename}` — joblib-serialized `GraphEdgeClassifier` (scikit-learn pipeline wrapping XGBoost).
- `manifest.json` — model metadata: hyperparameters, feature lists, and a SHA-256 checksum of the model file.

## Data

Trained on the curated PROTAC-Splitter dataset: https://doi.org/10.5281/zenodo.15797309
"""


@dataclasses.dataclass
class Args:
    """Push a trained XGBoost edge classifier to the HuggingFace Hub."""

    model_path: str = "./models/PROTAC-Splitter-XGBoost.joblib"
    """Path to the trained GraphEdgeClassifier .joblib file."""

    repo_id: str = "ailab-bio/PROTAC-Splitter-XGBoost"
    """HuggingFace Hub model repo to push to."""

    hub_token: str = ""
    """HuggingFace token (defaults to HF_TOKEN in .env)."""

    private: bool = False
    """Push as a private repo instead of public."""

    commit_message: str = "Upload PROTAC-Splitter XGBoost model"
    """Commit message for the upload."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return version("protac-splitter")
    except PackageNotFoundError:
        return "unknown"


def _build_manifest(model_path: Path, model: GraphEdgeClassifier) -> dict:
    clf = model.pipeline.named_steps.get("clf")
    return {
        "model_filename": model_path.name,
        "sha256": _sha256(model_path),
        "size_bytes": model_path.stat().st_size,
        "pushed_at": datetime.now(timezone.utc).isoformat(),
        "protac_splitter_version": _package_version(),
        "library": "xgboost",
        "wrapper_class": "protac_splitter.graphs.edge_classifier.GraphEdgeClassifier",
        "hyperparameters": {
            "binary": model.binary,
            "n_bits": model.n_bits,
            "radius": model.radius,
            "use_descriptors": model.use_descriptors,
            "use_fingerprints": model.use_fingerprints,
            "use_svd_fp": model.use_svd_fp,
            "n_svd_components": model.n_svd_components,
            "scaler_graph": model.scaler_graph,
            "scaler_desc": model.scaler_desc,
            "smote_k_neighbors": model.smote_k_neighbors,
        },
        "xgb_params": clf.get_params() if clf is not None else model.xgb_params,
        "features": {
            "graph_features": model.graph_features,
            "categorical_features": model.categorical_features,
            "descriptor_features": model.descriptor_features,
            "descriptor_names": model.descriptor_names,
            "fingerprint_features": model.fingerprint_features,
        },
    }


def main(args: Args) -> None:
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    token = get_hub_token(args.hub_token)
    model = GraphEdgeClassifier.load(model_path)
    manifest = _build_manifest(model_path, model)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        os.symlink(model_path.resolve(), tmp_dir / model_path.name)
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        (tmp_dir / "README.md").write_text(
            _MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id, model_filename=model_path.name)
        )

        print(f"Uploading {model_path}, manifest.json, and README.md to {args.repo_id}...")
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            commit_message=args.commit_message,
        )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
