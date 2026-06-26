import hashlib
import logging
import warnings
import requests
from pathlib import Path
from typing import Union, Optional, Dict, List, Literal

from rdkit import Chem
from datasets import Dataset
import pandas as pd

from protac_splitter.config import get_cache_dir, get_hf_token
from protac_splitter.chemoinformatics import canonize
from protac_splitter.evaluation import split_prediction
from protac_splitter.fixing_functions import fix_prediction
from protac_splitter.llms.model_utils import get_pipeline, run_pipeline
from protac_splitter.graphs.clustering import get_representative_e3s_fp
from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier
from protac_splitter.graphs.splitting_algorithms import split_protac_graph_based

_XGBOOST_MODEL_FILENAME = "PROTAC-Splitter-XGBoost.joblib"
_XGBOOST_DOWNLOAD_URL = (
    "https://zenodo.org/records/15797310/files/"
    "PROTAC-Splitter-XGBoost.joblib?download=1"
)
_XGBOOST_SHA256 = "513621f4dc2ff7ec819a222bc7311afb8b6e6e89d6d694dd2906e695a50086dd"

_VALID_MODELS = frozenset({
    "transformer",
    "xgboost",
    "heuristic",
    "transformer->xgboost",
    "xgboost->heuristic",
    "xgboost+heuristic",
    "heuristic->xgboost",
    "heuristic+xgboost",
})


def load_graph_edge_classifier_from_cache(
    cache_dir: Union[str, Path, None] = None,
    model_filename: str = _XGBOOST_MODEL_FILENAME,
    download_url: str = _XGBOOST_DOWNLOAD_URL,
) -> GraphEdgeClassifier:
    """Load the XGBoost GraphEdgeClassifier, downloading from Zenodo on first use.

    Args:
        cache_dir: Directory to cache the model. Defaults to ``get_cache_dir()``
            (controlled by ``PROTAC_SPLITTER_CACHE_DIR`` env var / .env file).
        model_filename: Filename to use inside ``cache_dir``.
        download_url: URL to download the model from if not already cached.

    Returns:
        GraphEdgeClassifier: Loaded classifier.
    """
    cache_path = Path(cache_dir).expanduser() if cache_dir is not None else get_cache_dir()
    cache_path.mkdir(parents=True, exist_ok=True)
    model_path = cache_path / model_filename

    if not model_path.exists():
        logging.info(f"Downloading XGBoost model → {model_path} ...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        expected_size = int(response.headers.get("Content-Length", -1))

        with model_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        if expected_size != -1:
            actual = model_path.stat().st_size
            if actual != expected_size:
                model_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download incomplete: got {actual} bytes, expected {expected_size}."
                )

        h = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if h != _XGBOOST_SHA256:
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded model checksum mismatch: got {h}, expected {_XGBOOST_SHA256}. "
                "The file has been removed — please try again."
            )
        logging.info("XGBoost model downloaded and verified.")

    return GraphEdgeClassifier.load(model_path)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _linker_heavy_atom_count(pred_str: Optional[str]) -> int:
    """Return the non-dummy heavy-atom count of the linker in an 'e3.linker.poi' string, or -1."""
    if pred_str is None:
        return -1
    parts = split_prediction(pred_str)
    if parts is None or parts.get("linker") is None:
        return -1
    mol = Chem.MolFromSmiles(parts["linker"])
    if mol is None:
        return -1
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 0)


def _pred_dict_to_str(pred: Optional[Dict[str, str]]) -> Optional[str]:
    """Convert a graph-split result dict to the dot-separated 'e3.linker.poi' string."""
    if pred is None or any(v is None for v in pred.values()):
        return None
    return f"{pred['e3']}.{pred['linker']}.{pred['poi']}"


def _resolve_model(
    use_transformer: Optional[bool],
    use_xgboost: Optional[bool],
    model: Optional[str],
) -> str:
    """Return the canonical model string, issuing a DeprecationWarning for legacy bool args."""
    if model is not None:
        return model
    if use_transformer is None and use_xgboost is None:
        return "xgboost"  # default behaviour
    warnings.warn(
        "use_transformer and use_xgboost are deprecated and will be removed in a future release. "
        "Use the `model` argument instead "
        "(e.g. model='xgboost', model='transformer->xgboost', model='heuristic').",
        DeprecationWarning,
        stacklevel=3,
    )
    xt = bool(use_transformer)
    xg = use_xgboost if use_xgboost is not None else True
    if xt and xg:
        return "transformer->xgboost"
    if xt:
        return "transformer"
    if xg:
        return "xgboost"
    return "heuristic"


def _smiles_to_dataset(
    protac_smiles: Union[str, List[str], pd.DataFrame],
    protac_smiles_col: str,
) -> Dataset:
    """Canonize and validate input SMILES, return a HuggingFace Dataset."""
    if isinstance(protac_smiles, str):
        canon = canonize(protac_smiles)
        if canon is None:
            raise ValueError(f"Invalid PROTAC SMILES: {protac_smiles}")
        return Dataset.from_dict({protac_smiles_col: [canon]})

    if isinstance(protac_smiles, list):
        canon_list = [canonize(s) for s in protac_smiles]
        invalid = [s for s, c in zip(protac_smiles, canon_list) if c is None]
        if invalid:
            raise ValueError(f"Invalid PROTAC SMILES in list: {invalid}")
        return Dataset.from_dict({protac_smiles_col: canon_list})

    if isinstance(protac_smiles, pd.DataFrame):
        if protac_smiles_col not in protac_smiles.columns:
            raise ValueError(f'DataFrame must contain a column named "{protac_smiles_col}".')
        canon_series = protac_smiles[protac_smiles_col].apply(canonize)
        if canon_series.isnull().any():
            raise ValueError(
                f"Invalid PROTAC SMILES in DataFrame: {protac_smiles[canon_series.isnull()]}"
            )
        return Dataset.from_pandas(canon_series.to_frame(name=protac_smiles_col))

    raise TypeError(f"protac_smiles must be str, list, or DataFrame; got {type(protac_smiles)}")


# ---------------------------------------------------------------------------
# Per-strategy runners — each returns a Dataset with columns:
#   [protac_smiles_col, "default_pred_n0", "model_name"]
# ---------------------------------------------------------------------------

def _run_xgboost_ds(
    ds: Dataset,
    xgboost_model: GraphEdgeClassifier,
    representative_e3s_fp: List,
    protac_smiles_col: str,
    betweenness_threshold: float,
    use_capacity_weight: bool,
    betweenness_approx_frac: Optional[float],
) -> Dataset:
    def _map(row):
        protac = row[protac_smiles_col]
        pred = split_protac_graph_based(
            protac_smiles=protac,
            use_classifier=True,
            classifier=xgboost_model,
            representative_e3s_fp=representative_e3s_fp,
            betweenness_threshold=betweenness_threshold,
            use_capacity_weight=use_capacity_weight,
            betweenness_approx_frac=betweenness_approx_frac,
        )
        return {protac_smiles_col: protac, "default_pred_n0": _pred_dict_to_str(pred), "model_name": "XGBoost"}

    return ds.map(_map, num_proc=1, desc="Splitting with XGBoost")


def _run_heuristic_ds(
    ds: Dataset,
    protac_smiles_col: str,
    betweenness_threshold: float,
    use_capacity_weight: bool,
    betweenness_approx_frac: Optional[float],
    num_proc: int = 1,
    verbose: int = 0,
) -> Dataset:
    def _map(row):
        protac = row[protac_smiles_col]
        pred = split_protac_graph_based(
            protac_smiles=protac,
            use_classifier=False,
            betweenness_threshold=betweenness_threshold,
            use_capacity_weight=use_capacity_weight,
            betweenness_approx_frac=betweenness_approx_frac,
            verbose=verbose,
        )
        return {protac_smiles_col: protac, "default_pred_n0": _pred_dict_to_str(pred), "model_name": "Heuristic"}

    return ds.map(_map, num_proc=num_proc, desc="Splitting with Heuristic")


def _run_xgboost_then_heuristic_ds(
    ds: Dataset,
    xgboost_model: GraphEdgeClassifier,
    representative_e3s_fp: List,
    protac_smiles_col: str,
    betweenness_threshold: float,
    use_capacity_weight: bool,
    betweenness_approx_frac: Optional[float],
    verbose: int = 0,
) -> Dataset:
    def _map(row):
        protac = row[protac_smiles_col]
        pred = split_protac_graph_based(
            protac_smiles=protac,
            use_classifier=True,
            classifier=xgboost_model,
            representative_e3s_fp=representative_e3s_fp,
            betweenness_threshold=betweenness_threshold,
            use_capacity_weight=use_capacity_weight,
            betweenness_approx_frac=betweenness_approx_frac,
            verbose=verbose,
        )
        split = _pred_dict_to_str(pred)
        if split is None:
            pred = split_protac_graph_based(
                protac_smiles=protac,
                use_classifier=False,
                betweenness_threshold=betweenness_threshold,
                use_capacity_weight=use_capacity_weight,
                betweenness_approx_frac=betweenness_approx_frac,
                verbose=verbose,
            )
            split = _pred_dict_to_str(pred)
            model_name = "Heuristic"
        else:
            model_name = "XGBoost"
        return {protac_smiles_col: protac, "default_pred_n0": split, "model_name": model_name}

    return ds.map(_map, num_proc=1, desc="Splitting with XGBoost → Heuristic fallback")

def _run_heuristic_then_xgboost_ds(
    ds: Dataset,
    xgboost_model: GraphEdgeClassifier,
    representative_e3s_fp: List,
    protac_smiles_col: str,
    betweenness_threshold: float,
    use_capacity_weight: bool,
    betweenness_approx_frac: Optional[float],
    num_proc: int = 1,
    verbose: int = 0,
) -> Dataset:
    # Phase 1: heuristic on all rows — parallelisable.
    result_ds = _run_heuristic_ds(
        ds, protac_smiles_col, betweenness_threshold,
        use_capacity_weight, betweenness_approx_frac,
        num_proc=num_proc, verbose=verbose,
    )

    failed = [i for i, r in enumerate(result_ds) if r["default_pred_n0"] is None]
    if not failed:
        return result_ds

    # Phase 2: XGBoost on failures only — single-process (not thread-safe).
    xgb_ds = _run_xgboost_ds(
        ds.select(failed), xgboost_model, representative_e3s_fp, protac_smiles_col,
        betweenness_threshold, use_capacity_weight, betweenness_approx_frac,
    )

    result_df = result_ds.to_pandas()
    result_df.iloc[failed] = xgb_ds.to_pandas().values
    return Dataset.from_pandas(result_df, preserve_index=False)


def _run_xgboost_and_heuristic_ds(
    ds: Dataset,
    xgboost_model: GraphEdgeClassifier,
    representative_e3s_fp: List,
    protac_smiles_col: str,
    betweenness_threshold: float,
    use_capacity_weight: bool,
    betweenness_approx_frac: Optional[float],
    num_proc: int = 1,
    verbose: int = 0,
) -> Dataset:
    # Phase 1: heuristic on all rows in parallel.
    heuristic_ds = _run_heuristic_ds(
        ds, protac_smiles_col, betweenness_threshold,
        use_capacity_weight, betweenness_approx_frac,
        num_proc=num_proc, verbose=verbose,
    )
    # Phase 2: XGBoost on all rows — single-process (not thread-safe).
    xgb_ds = _run_xgboost_ds(
        ds, xgboost_model, representative_e3s_fp, protac_smiles_col,
        betweenness_threshold, use_capacity_weight, betweenness_approx_frac,
    )

    # Merge both predictions into a combined dataset for parallel selection.
    merged_df = heuristic_ds.to_pandas().rename(columns={
        "default_pred_n0": "_heuristic_pred",
        "model_name": "_heuristic_model",
    })
    merged_df["_xgb_pred"] = xgb_ds.to_pandas()["default_pred_n0"].values
    merged_ds = Dataset.from_pandas(merged_df, preserve_index=False)

    def _select(row):
        xgb_atoms = _linker_heavy_atom_count(row["_xgb_pred"])
        heuristic_atoms = _linker_heavy_atom_count(row["_heuristic_pred"])
        if xgb_atoms >= heuristic_atoms:
            return {"default_pred_n0": row["_xgb_pred"], "model_name": "XGBoost"}
        return {"default_pred_n0": row["_heuristic_pred"], "model_name": "Heuristic"}

    return merged_ds.map(
        _select,
        num_proc=num_proc,
        remove_columns=["_heuristic_pred", "_heuristic_model", "_xgb_pred"],
        desc="Selecting best prediction (longest linker)",
    )


def _run_transformer_ds(
    ds: Dataset,
    pipe,
    batch_size: int,
    beam_size: int,
    protac_smiles_col: str,
    fix_predictions: bool,
    verbose: int,
    num_proc: int,
    xgboost_fallback: bool = False,
    xgboost_model: Optional[GraphEdgeClassifier] = None,
    representative_e3s_fp: Optional[List] = None,
    betweenness_threshold: float = 0.4,
    use_capacity_weight: bool = False,
    betweenness_approx_frac: Optional[float] = None,
) -> Dataset:
    raw_preds = run_pipeline(
        pipe,
        ds,
        batch_size,
        is_causal_language_model=False,
        smiles_column=protac_smiles_col,
    )
    preds_df = pd.DataFrame(raw_preds)
    preds_df[protac_smiles_col] = ds[protac_smiles_col]
    preds_ds = Dataset.from_pandas(preds_df)

    def _map(row):
        protac = row[protac_smiles_col]
        beam_preds = {
            k: (fix_prediction(protac, v, verbose=verbose) if fix_predictions else v)
            for k, v in row.items()
            if k.startswith("pred_")
        }
        if all(v is None for v in beam_preds.values()):
            if xgboost_fallback and xgboost_model is not None:
                pred = split_protac_graph_based(
                    protac_smiles=protac,
                    use_classifier=True,
                    classifier=xgboost_model,
                    representative_e3s_fp=representative_e3s_fp,
                    betweenness_threshold=betweenness_threshold,
                    use_capacity_weight=use_capacity_weight,
                    betweenness_approx_frac=betweenness_approx_frac,
                )
                return {
                    protac_smiles_col: protac,
                    "default_pred_n0": _pred_dict_to_str(pred),
                    "model_name": "XGBoost",
                }
            return {protac_smiles_col: protac, "default_pred_n0": None, "model_name": "Transformer"}

        for i in range(beam_size):
            v = beam_preds.get(f"pred_n{i}")
            if v is not None:
                return {protac_smiles_col: protac, "default_pred_n0": v, "model_name": "Transformer"}

        return {protac_smiles_col: protac, "default_pred_n0": None, "model_name": "Transformer"}

    desc_parts = []
    if fix_predictions:
        desc_parts.append("Fixing predictions")
    if xgboost_fallback:
        desc_parts.append("XGBoost fallback")
    return preds_ds.map(
        _map,
        # XGBoost inside map is not thread-safe
        num_proc=1 if xgboost_fallback else num_proc,
        desc=" and ".join(desc_parts) or "Selecting best Transformer prediction",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_protac(
        protac_smiles: Union[str, List, pd.DataFrame],
        use_transformer: Optional[bool] = None,
        use_xgboost: Optional[bool] = None,
        fix_predictions: bool = True,
        protac_smiles_col: str = "SMILES",
        batch_size: int = 1,
        beam_size: int = 5,
        device: Optional[Union[int, str]] = None,
        num_proc: int = 1,
        verbose: int = 0,
        betweenness_threshold: float = 0.4,
        use_capacity_weight: bool = False,
        betweenness_approx_frac: float = None,
        model: Optional[Literal[
            "transformer",
            "xgboost",
            "heuristic",
            "transformer->xgboost",
            "xgboost->heuristic",
            "xgboost+heuristic",
            "heuristic->xgboost",
            "heuristic+xgboost",
        ]] = None,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """Split a PROTAC SMILES into E3 ligand, linker, and POI warhead.

    The splitting strategy is controlled by the ``model`` argument. Supported values:

    +-------------------------------+-------------------------------------------------------+
    | ``model`` value               | Description                                           |
    +===============================+=======================================================+
    | ``"xgboost"`` *(default)*     | XGBoost graph edge-classifier. No GPU required.       |
    |                               | Model (~17 MB) downloaded automatically on first use. |
    +-------------------------------+-------------------------------------------------------+
    | ``"heuristic"``               | Betweenness-centrality algorithm. No download needed. |
    +-------------------------------+-------------------------------------------------------+
    | ``"transformer"``             | Seq2seq Transformer (``ailab-bio/PROTAC-Splitter``).  |
    |                               | Requires the ``[transformer]`` extra; GPU recommended.|
    +-------------------------------+-------------------------------------------------------+
    | ``"transformer->xgboost"``    | Transformer first; XGBoost replaces failed results.   |
    +-------------------------------+-------------------------------------------------------+
    | ``"xgboost->heuristic"``      | XGBoost first; heuristic replaces failed results.     |
    +-------------------------------+-------------------------------------------------------+
    | ``"heuristic->xgboost"``      | Heuristic first; XGBoost replaces failed results.     |
    +-------------------------------+-------------------------------------------------------+
    | ``"xgboost+heuristic"``       | Placeholder — not yet implemented.                    |
    +-------------------------------+-------------------------------------------------------+
    | ``"heuristic+xgboost"``       | Reserved for future use.                              |
    +-------------------------------+-------------------------------------------------------+

    Args:
        protac_smiles: SMILES to split. Accepts a single string, a list of strings,
            or a DataFrame with a ``protac_smiles_col`` column.
        model: Splitting strategy. See the table above for valid values. Takes
            precedence over the deprecated ``use_transformer`` / ``use_xgboost``
            flags. Default ``None`` resolves to ``"xgboost"``.
        use_transformer: *Deprecated.* Use ``model='transformer'`` or
            ``model='transformer->xgboost'`` instead. Setting this to ``True`` maps to
            ``model='transformer->xgboost'`` when ``use_xgboost`` is also ``True``, and
            to ``model='transformer'`` otherwise.
        use_xgboost: *Deprecated.* Use ``model='xgboost'`` instead. Setting this to
            ``False`` while ``use_transformer`` is also ``False`` maps to
            ``model='heuristic'``.
        fix_predictions: Apply deterministic cheminformatics corrections to Transformer
            predictions before checking reassembly. Only used when the Transformer is
            active. Default ``True``.
        protac_smiles_col: Column name for SMILES when the input is a DataFrame, and
            output key name in result dicts. Default ``"SMILES"``.
        batch_size: Inference batch size for the Transformer. Default ``1``.
        beam_size: Number of beam-search candidates from the Transformer. Higher values
            may improve quality at the cost of speed. Default ``5``.
        device: Device for the Transformer model (e.g. ``0``, ``"cuda"``, ``"cpu"``).
            ``None`` auto-selects GPU when available.
        num_proc: Worker processes for parallel dataset mapping. Only effective for the
            heuristic strategy (XGBoost mapping is always single-process). Default ``1``.
        verbose: Verbosity level passed to fixing functions. Default ``0``.
        betweenness_threshold: Betweenness-centrality cut-off for the heuristic
            algorithm. Higher values are more conservative. Default ``0.4``.
        use_capacity_weight: Weight graph edges by bond capacity when computing
            betweenness centrality (heuristic only). Default ``False``.
        betweenness_approx_frac: Fraction of nodes to sample for approximate
            betweenness centrality. ``None`` uses exact computation. Default ``None``.

    Returns:
        * Single string input → ``dict`` with keys ``protac_smiles_col``,
          ``"default_pred_n0"`` (``"e3.linker.poi"`` format), and ``"model_name"``.
        * List input → list of such dicts.
        * DataFrame input → DataFrame with the same columns.
    """
    model_str = _resolve_model(use_transformer, use_xgboost, model)
    if model_str not in _VALID_MODELS:
        raise ValueError(
            f"`model` must be one of {sorted(_VALID_MODELS)}. Got: {model_str!r}"
        )

    # Load required resources up-front so errors surface before any SMILES processing.
    needs_xgboost = "xgboost" in model_str
    needs_transformer = "transformer" in model_str

    if needs_xgboost:
        representative_e3s_fp = get_representative_e3s_fp()
        xgboost_model = load_graph_edge_classifier_from_cache()
    else:
        representative_e3s_fp = None
        xgboost_model = None

    if needs_transformer:
        pipe = get_pipeline(
            model_name="ailab-bio/PROTAC-Splitter",
            token=get_hf_token(),
            is_causal_language_model=False,
            num_return_sequences=beam_size,
            device=device,
        )
    else:
        pipe = None

    ds = _smiles_to_dataset(protac_smiles, protac_smiles_col)

    _graph_kwargs = dict(
        betweenness_threshold=betweenness_threshold,
        use_capacity_weight=use_capacity_weight,
        betweenness_approx_frac=betweenness_approx_frac,
    )

    if model_str == "transformer":
        preds_ds = _run_transformer_ds(
            ds, pipe, batch_size, beam_size, protac_smiles_col,
            fix_predictions, verbose, num_proc,
        )
    elif model_str == "transformer->xgboost":
        preds_ds = _run_transformer_ds(
            ds, pipe, batch_size, beam_size, protac_smiles_col,
            fix_predictions, verbose, num_proc,
            xgboost_fallback=True,
            xgboost_model=xgboost_model,
            representative_e3s_fp=representative_e3s_fp,
            **_graph_kwargs,
        )
    elif model_str == "xgboost":
        preds_ds = _run_xgboost_ds(
            ds, xgboost_model, representative_e3s_fp, protac_smiles_col,
            **_graph_kwargs,
        )
    elif model_str == "xgboost->heuristic":
        preds_ds = _run_xgboost_then_heuristic_ds(
            ds, xgboost_model, representative_e3s_fp, protac_smiles_col,
            **_graph_kwargs, verbose=verbose,
        )
    elif model_str == "heuristic->xgboost":
        preds_ds = _run_heuristic_then_xgboost_ds(
            ds, xgboost_model, representative_e3s_fp, protac_smiles_col,
            **_graph_kwargs, num_proc=num_proc, verbose=verbose,
        )
    elif model_str == "xgboost+heuristic" or model_str == "heuristic+xgboost":
        preds_ds = _run_xgboost_and_heuristic_ds(
            ds, xgboost_model, representative_e3s_fp, protac_smiles_col,
            **_graph_kwargs, num_proc=num_proc, verbose=verbose,
        )
    elif model_str == "heuristic":
        preds_ds = _run_heuristic_ds(
            ds, protac_smiles_col, **_graph_kwargs, num_proc=num_proc, verbose=verbose,
        )
    else:
        raise NotImplementedError(
            f"model='{model_str}' is not yet implemented. "
            "Contributions welcome — see the _run_*_ds helpers in protac_splitter.py."
        )

    if isinstance(protac_smiles, str):
        return preds_ds[0]
    if isinstance(protac_smiles, pd.DataFrame):
        return preds_ds.to_pandas()
    return list(preds_ds)
