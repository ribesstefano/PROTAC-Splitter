"""``ChemPropSplitter``: a ChemProp bond classifier that splits PROTACs, with HuggingFace Hub I/O.

On the Hub a model is two files: ``config.json`` (architecture) and ``model.pt``, a
plain tensor ``state_dict`` loaded with ``torch.load(weights_only=True)``, so loading
never unpickles code. Lightning training checkpoints (``best.ckpt``) are converted
with :meth:`ChemPropSplitter.from_checkpoint`.
"""
import io
import json
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from protac_splitter.chemprop.inference import (
    BOND_PREDICTOR_KEYS,
    MESSAGE_PASSING_KEYS,
    build_model,
    decode_pair,
    parse_protac,
    predict_arrays,
)

DEFAULT_REPO_ID = "ailab-bio/PROTAC-Splitter-ChemProp"
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.pt"

# Only these submodules carry weights used at inference; the rest of a Lightning
# state_dict is metric/loss buffers rebuilt by build_model.
_WEIGHT_PREFIXES = ("message_passing.", "bond_predictor.")


def _resolve_device(device: Optional[Union[int, str]]) -> str:
    import torch
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int):
        return f"cuda:{device}" if device >= 0 else "cpu"
    return device


class _CheckpointUnpickler(pickle.Unpickler):
    """Resolves the training script's top-level ``model_utils`` module to this package's copy."""

    def find_class(self, module, name):
        if module == "model_utils":
            module = "protac_splitter.chemprop.model_utils"
        return super().find_class(module, name)


class _CheckpointPickle:
    """``pickle_module`` for ``torch.load`` that uses :class:`_CheckpointUnpickler`."""
    Unpickler = _CheckpointUnpickler
    load = staticmethod(lambda f, **kw: _CheckpointUnpickler(f, **kw).load())
    loads = staticmethod(lambda b, **kw: _CheckpointUnpickler(io.BytesIO(b), **kw).load())


class ChemPropSplitter:
    """Split PROTACs into E3 ligand, linker, and POI warhead with a ChemProp bond classifier.

    Example:
        >>> splitter = ChemPropSplitter.from_pretrained()
        >>> splitter.split(["<PROTAC SMILES>"])
        ['<e3>.<linker>.<poi>']
    """

    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Loading and saving
    # ------------------------------------------------------------------

    @classmethod
    def from_state_dict(cls, config: Dict, state_dict: Dict, device: Optional[Union[int, str]] = None):
        model = build_model(config)
        weights = {k: v for k, v in state_dict.items() if k.startswith(_WEIGHT_PREFIXES)}
        missing, unexpected = model.load_state_dict(weights, strict=False)
        missing = [k for k in missing if k.startswith(_WEIGHT_PREFIXES)]
        if missing or unexpected:
            raise RuntimeError(
                f"Weights do not match the config: missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        model.to(_resolve_device(device))
        return cls(model, config)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], device: Optional[Union[int, str]] = None):
        """Convert a Lightning ``best.ckpt`` written by the ChemProp training pipeline.

        A Lightning checkpoint is a full pickle, so only load your own/trusted files.
        """
        import torch
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False,
                                pickle_module=_CheckpointPickle)
        hp = checkpoint["hyper_parameters"]
        state_dict = checkpoint["state_dict"]
        config = {
            "message_passing": {k: hp["message_passing"][k] for k in MESSAGE_PASSING_KEYS},
            "bond_predictor": {k: hp["bond_predictor"][k] for k in BOND_PREDICTOR_KEYS},
            "class_weights": state_dict["bond_predictor.criterion.class_weights"].tolist(),
        }
        return cls.from_state_dict(config, state_dict, device=device)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Write ``config.json`` and ``model.pt`` to ``save_directory``."""
        import torch
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / CONFIG_FILENAME).write_text(json.dumps(self.config, indent=2) + "\n")
        weights = {k: v.detach().cpu().contiguous()
                   for k, v in self.model.state_dict().items() if k.startswith(_WEIGHT_PREFIXES)}
        torch.save(weights, save_directory / WEIGHTS_FILENAME)

    @classmethod
    def load(cls, directory: Union[str, Path], device: Optional[Union[int, str]] = None):
        """Load a model written by :meth:`save_pretrained`."""
        import torch
        directory = Path(directory)
        config = json.loads((directory / CONFIG_FILENAME).read_text())
        state_dict = torch.load(directory / WEIGHTS_FILENAME, map_location="cpu", weights_only=True)
        return cls.from_state_dict(config, state_dict, device=device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        device: Optional[Union[int, str]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
    ) -> "ChemPropSplitter":
        """Load a ChemPropSplitter from the HuggingFace Hub (or a local directory).

        Files are downloaded (and cached for subsequent calls) via ``huggingface_hub``
        into the standard HuggingFace Hub cache, like the XGBoost and Transformer models.

        Args:
            repo_id: HuggingFace Hub model repo, or a local directory written by
                :meth:`save_pretrained`.
            device: Torch device (e.g. ``"cpu"``, ``"cuda"``, ``0``). ``None`` picks
                CUDA when available.
            cache_dir: Directory to cache the download in. Defaults to the standard
                HuggingFace Hub cache (``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE``).
            token: HuggingFace token, for private repos. Defaults to no auth.
            revision: Git revision (branch, tag, or commit SHA) to load.
            force_download: Re-download even if a cached copy already exists.
        """
        if Path(repo_id).is_dir():
            return cls.load(repo_id, device=device)
        from huggingface_hub import hf_hub_download
        paths = [
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                token=token,
                revision=revision,
                force_download=force_download,
            )
            for filename in (CONFIG_FILENAME, WEIGHTS_FILENAME)
        ]
        return cls.load(Path(paths[0]).parent, device=device)

    def push_to_hub(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload PROTAC-Splitter ChemProp model",
        extra_files: Optional[Dict[str, Union[str, bytes]]] = None,
    ) -> str:
        """Upload ``config.json`` and ``model.pt`` (plus ``extra_files``) as one commit.

        Args:
            repo_id: HuggingFace Hub model repo; created if missing.
            token: HuggingFace token with write access.
            private: Create the repo as private. Ignored if the repo already exists.
            commit_message: Commit message for the upload.
            extra_files: ``{filename: content}`` to upload alongside, e.g. a model card
                as ``README.md``.

        Returns:
            The URL of the commit.
        """
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            for filename, content in (extra_files or {}).items():
                mode = "wb" if isinstance(content, bytes) else "w"
                with open(Path(tmp_dir) / filename, mode) as f:
                    f.write(content)
            commit = api.upload_folder(
                folder_path=tmp_dir, repo_id=repo_id, repo_type="model",
                commit_message=commit_message,
            )
        return commit.commit_url

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, smiles: Sequence[str], batch_size: int = 64) -> List[Optional[np.ndarray]]:
        """Bond-class probabilities ``(n_bonds, 3)`` per SMILES, in RDKit bond order of
        ``Chem.MolFromSmiles(smiles)``; None for SMILES the model cannot take."""
        mols = [parse_protac(s) for s in smiles]
        valid = [i for i, mol in enumerate(mols) if mol is not None]
        probs = predict_arrays(self.model, [mols[i] for i in valid], batch_size)
        result: List[Optional[np.ndarray]] = [None] * len(mols)
        for i, p in zip(valid, probs):
            result[i] = p
        return result

    def split(self, smiles: Sequence[str], batch_size: int = 64) -> List[Optional[str]]:
        """Split each SMILES into an ``"e3.linker.poi"`` string, or None when it cannot be split.

        Attachment points are ``[*:1]`` (POI side) and ``[*:2]`` (E3 side).
        """
        result: List[Optional[str]] = []
        for smi, probs in zip(smiles, self.predict_proba(smiles, batch_size)):
            pair = decode_pair(parse_protac(smi), probs) if probs is not None else None
            if pair is None:
                result.append(None)
                continue
            parts = pair["parts"]
            # part_1 carries [*:1] (POI side), part_2 carries [*:2] (E3 side).
            result.append(f"{parts['part_2']}.{parts['linker']}.{parts['part_1']}")
        return result
