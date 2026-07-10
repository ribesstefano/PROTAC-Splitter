import hashlib
import logging
import warnings
from pathlib import Path
from typing import Union, Optional, Dict, List, Literal, Tuple

# Import first, before numpy-linked packages (rdkit/datasets/pandas below): this sets
# thread-count env vars for native math libraries (OpenMP/OpenBLAS/MKL/Accelerate),
# which some of them only read once, at first load. Importing it after would be too
# late and leave those libraries free to over-provision threads on cgroup-limited
# containers.
from protac_splitter.config import get_cache_dir, get_hf_token

import requests
from rdkit import Chem
from datasets import Dataset
import pandas as pd

from protac_splitter.chemoinformatics import canonize
from protac_splitter.evaluation import split_prediction
from protac_splitter.fixing_functions import fix_prediction
from protac_splitter.graphs.clustering import get_representative_e3s_fp, get_representative_whs_fp
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
    "adaptive",
})

# (betweenness_threshold, use_capacity_weight) combinations tried by model="adaptive"'s
# heuristic stage, in order — the package default is first so the common/easy case
# (default already gives a flag-free split) costs exactly one heuristic call.
_DEFAULT_ADAPTIVE_GRID: List[Tuple[float, bool]] = [
    (0.4, False),
    (0.3, False),
    (0.5, False),
    (0.4, True),
    (0.3, True),
    (0.5, True),
]


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
        _download_xgboost_model(download_url, model_path)

    return GraphEdgeClassifier.load(model_path)


def _download_xgboost_model(
    download_url: str,
    model_path: Path,
    num_attempts: int = 3,
    connect_timeout: float = 10.0,
    read_timeout: float = 30.0,
) -> None:
    """Download the XGBoost model to `model_path`, retrying on transient network errors.

    Downloads to a temporary file first and only renames it into place once fully
    verified, so a killed or timed-out download never leaves a corrupt file behind
    that a later run would mistake for a valid cache hit.
    """
    tmp_path = model_path.with_suffix(model_path.suffix + ".part")
    last_error: Optional[Exception] = None

    for attempt in range(1, num_attempts + 1):
        try:
            logging.info(f"Downloading XGBoost model → {model_path} (attempt {attempt}/{num_attempts}) ...")
            response = requests.get(download_url, stream=True, timeout=(connect_timeout, read_timeout))
            response.raise_for_status()
            expected_size = int(response.headers.get("Content-Length", -1))

            with tmp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            if expected_size != -1:
                actual = tmp_path.stat().st_size
                if actual != expected_size:
                    raise RuntimeError(
                        f"Download incomplete: got {actual} bytes, expected {expected_size}."
                    )

            h = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
            if h != _XGBOOST_SHA256:
                raise RuntimeError(
                    f"Downloaded model checksum mismatch: got {h}, expected {_XGBOOST_SHA256}."
                )

            tmp_path.rename(model_path)
            logging.info("XGBoost model downloaded and verified.")
            return
        except (requests.exceptions.RequestException, RuntimeError) as e:
            last_error = e
            tmp_path.unlink(missing_ok=True)
            logging.warning(f"XGBoost model download attempt {attempt}/{num_attempts} failed: {e}")

    raise RuntimeError(
        f"Failed to download the XGBoost model from {download_url} after {num_attempts} attempts: "
        f"{last_error}. You can also download it manually and place it at {model_path} "
        "(or point PROTAC_SPLITTER_CACHE_DIR at a directory that already contains it)."
    ) from last_error


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _safe_num_proc(num_proc: Optional[int]) -> Optional[int]:
    """Normalize a num_proc value for passing to Dataset.map().

    datasets.Dataset.map() only runs in-process when num_proc is None — passing 1
    (rather than omitting it) still forks a worker via multiprocessing.Pool, which is
    pure overhead for "1 worker" and, worse, can deadlock if a native library (e.g.
    XGBoost/OpenBLAS) already initialized threads in the parent before the fork.
    """
    return None if num_proc is None or num_proc <= 1 else num_proc


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
    betweenness_approx_frac: Optional[float] = None,
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

    # num_proc must stay None (not 1): datasets.Dataset.map() only runs in-process
    # when num_proc is None — passing 1 still forks a worker via multiprocessing.Pool,
    # which deadlocks if XGBoost/OpenBLAS already initialized native threads in the
    # parent (during model load, just above) before the fork.
    return ds.map(_map, desc="Splitting with XGBoost")


def _run_heuristic_ds(
    ds: Dataset,
    protac_smiles_col: str,
    betweenness_threshold: float,
    use_capacity_weight: bool,
    betweenness_approx_frac: Optional[float] = None,
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

    return ds.map(_map, num_proc=_safe_num_proc(num_proc), desc="Splitting with Heuristic")


def _run_xgboost_then_heuristic_ds(
    ds: Dataset,
    xgboost_model: GraphEdgeClassifier,
    representative_e3s_fp: List,
    protac_smiles_col: str,
    betweenness_threshold: float,
    use_capacity_weight: bool,
    betweenness_approx_frac: Optional[float] = None,
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

    # See the num_proc note in _run_xgboost_ds above — must be None, not 1.
    return ds.map(_map, desc="Splitting with XGBoost → Heuristic fallback")

def _run_heuristic_then_xgboost_ds(
    ds: Dataset,
    xgboost_model: GraphEdgeClassifier,
    representative_e3s_fp: List,
    protac_smiles_col: str,
    betweenness_threshold: float,
    use_capacity_weight: bool,
    betweenness_approx_frac: Optional[float] = None,
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
    betweenness_approx_frac: Optional[float] = None,
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
        num_proc=_safe_num_proc(num_proc),
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
    from protac_splitter.llms.model_utils import run_pipeline
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
        # XGBoost inside map is not thread-safe, so no true parallelism here — but
        # that means None, not 1: see the num_proc note in _run_xgboost_ds above.
        num_proc=None if xgboost_fallback else _safe_num_proc(num_proc),
        desc=" and ".join(desc_parts) or "Selecting best Transformer prediction",
    )


# ---------------------------------------------------------------------------
# model="adaptive" — QC-gated escalation.
#
# Each candidate split is scored with evaluation.score_split, the same
# reference-free plausibility checks (structural validity, fragment size, linker
# topology, known-ligand similarity) used by the offline dataset_qc.qc_row pass.
# Escalation is driven by that score rather than by outright failure alone, so a
# structurally-valid-but-implausible split (e.g. E3/POI roles swapped) still
# triggers a retry with a different method/parameters.
# ---------------------------------------------------------------------------

# Heuristic and XGBoost both cut real bonds and then assign E3-vs-POI identity via the
# same fingerprint-based comparison score_split itself uses; the Transformer's
# [*:1]/[*:2] labels come straight out of the sequence model with no such check, so on
# an exact tie its similarity score is treated as weaker evidence, not equal evidence.
_METHOD_PRIORITY: Dict[str, int] = {"Heuristic": 0, "XGBoost": 0, "Transformer": 1}


def _rank_key(score: Dict[str, object], n_flags: int, model_name: str) -> Tuple[bool, int, int, float]:
    """Lower is better: gate failure first, then flag count, then method priority
    (prefer Heuristic/XGBoost over the Transformer — see _METHOD_PRIORITY), then
    (negated) summed similarity to known E3/warhead ligands as a final tie-break.
    """
    similarity = (score.get("e3_sim_to_known_e3") or 0.0) + (score.get("poi_sim_to_known_wh") or 0.0)
    return (
        bool(score.get("flag_structural", True)),
        n_flags,
        _METHOD_PRIORITY.get(model_name, 0),
        -similarity,
    )


def _swap_e3_poi_labels(pred: str) -> str:
    """Swap [*:1] (POI) and [*:2] (E3) attachment labels throughout an
    'e3.linker.poi'-formatted prediction, producing the exact same bond cuts with
    E3/POI identity flipped. Uses a sentinel so the two replacements can't collide.
    """
    return (
        pred.replace("[*:1]", "\0")
        .replace("[*:2]", "[*:1]")
        .replace("\0", "[*:2]")
    )


def _best_orientation(
    protac: str,
    pred: Optional[str],
    representative_e3s_fp: List,
    representative_whs_fp: List,
    model_name: str,
) -> Tuple[Optional[str], Tuple[bool, int, int, float], int, str]:
    """Score `pred` as generated, and — only when its own score_split flags the roles as
    suspicious (flag_role_swap_suspected) — also try it with E3/POI swapped (same bond
    cuts), keeping whichever scores better under `_rank_key`. Returns the winning
    prediction along with its key (so callers comparing across candidates don't need to
    re-score it) and its (n_flags, reasons).

    Heuristic and XGBoost both assign E3-vs-POI identity via distinguish_fragments(),
    the same fingerprint-similarity heuristic score_split's own known-ligand-similarity
    check uses — so it can pick the right bonds but the wrong label. This deliberately
    does *not* just try both orientations unconditionally and keep whichever has fewer
    flags: checked against Datasets/smiles/dataset-curated-held-out.csv, that would
    "improve" ~15% of already-correctly-labeled rows, purely from chance threshold
    crossings on an unrelated flag (typically flag_poi_low_similarity/flag_e3_low_
    similarity, not anything about the roles). Gating on flag_role_swap_suspected —
    which already requires *both* cross-similarities to clear the noise floor, not just
    be the larger of two low numbers — cuts that down to ~3.5%, which is that flag's
    own baseline false-positive rate, not something this swap step adds on top of it.
    """
    from protac_splitter.evaluation import count_flags, score_split

    score = score_split(
        protac, pred,
        representative_e3s_fp=representative_e3s_fp, representative_whs_fp=representative_whs_fp,
    )
    n_flags, reasons = count_flags(score)
    key = _rank_key(score, n_flags, model_name)

    if pred is None or not score.get("flag_role_swap_suspected", False):
        return pred, key, n_flags, reasons

    swapped_pred = _swap_e3_poi_labels(pred)
    swapped_score = score_split(
        protac, swapped_pred,
        representative_e3s_fp=representative_e3s_fp, representative_whs_fp=representative_whs_fp,
    )
    swapped_n_flags, swapped_reasons = count_flags(swapped_score)
    swapped_key = _rank_key(swapped_score, swapped_n_flags, model_name)

    if swapped_key < key:
        return swapped_pred, swapped_key, swapped_n_flags, swapped_reasons
    return pred, key, n_flags, reasons


def _run_heuristic_grid_ds(
    ds: Dataset,
    protac_smiles_col: str,
    grid: List[Tuple[float, bool]],
    representative_e3s_fp: List,
    representative_whs_fp: List,
    betweenness_approx_frac: Optional[float],
    num_proc: int,
    verbose: int,
) -> Dataset:
    """Try each (betweenness_threshold, use_capacity_weight) in `grid`, in order,
    scoring every candidate (both as generated and with E3/POI roles swapped, via
    `_best_orientation`) with `score_split`. Keeps the best-scoring candidate,
    short-circuiting on the first flag-free one so the common/easy case (the default
    grid point already gives a clean split) doesn't pay for the rest of the grid.
    """
    def _map(row):
        protac = row[protac_smiles_col]
        best_pred, best_key, best_flags, best_params = None, None, (0, "no_candidate"), None
        for threshold, use_capacity in grid:
            pred = split_protac_graph_based(
                protac_smiles=protac,
                use_classifier=False,
                representative_e3s_fp=representative_e3s_fp,
                betweenness_threshold=threshold,
                use_capacity_weight=use_capacity,
                betweenness_approx_frac=betweenness_approx_frac,
                verbose=verbose,
            )
            pred_str = _pred_dict_to_str(pred)
            pred_str, key, n_flags, reasons = _best_orientation(
                protac, pred_str, representative_e3s_fp, representative_whs_fp, "Heuristic",
            )
            if best_key is None or key < best_key:
                best_key = key
                best_pred = pred_str
                best_flags = (n_flags, reasons)
                best_params = f"betweenness_threshold={threshold},use_capacity_weight={use_capacity}"
            if key[0] is False and key[1] == 0:
                break
        n_flags, reasons = best_flags
        return {
            protac_smiles_col: protac,
            "default_pred_n0": best_pred,
            "model_name": "Heuristic",
            "heuristic_params": best_params,
            "n_flags": n_flags,
            "review_reasons": reasons,
        }

    return ds.map(_map, num_proc=_safe_num_proc(num_proc), desc="Heuristic grid search (QC-gated)")


def _run_transformer_scored_ds(
    ds: Dataset,
    pipe,
    batch_size: int,
    beam_size: int,
    protac_smiles_col: str,
    fix_predictions: bool,
    verbose: int,
    representative_e3s_fp: List,
    representative_whs_fp: List,
) -> Dataset:
    """Run the Transformer and keep whichever beam scores best under `score_split`,
    instead of just the first beam that happens to reassemble (contrast with
    _run_transformer_ds, used by the non-adaptive "transformer" strategies).
    """
    from protac_splitter.llms.model_utils import run_pipeline
    from protac_splitter.evaluation import count_flags, score_split

    raw_preds = run_pipeline(
        pipe, ds, batch_size, is_causal_language_model=False, smiles_column=protac_smiles_col,
    )
    preds_df = pd.DataFrame(raw_preds)
    preds_df[protac_smiles_col] = ds[protac_smiles_col]
    preds_ds = Dataset.from_pandas(preds_df)

    def _map(row):
        protac = row[protac_smiles_col]
        best_pred, best_key, best_flags = None, None, (0, "no_valid_beam")
        for i in range(beam_size):
            v = row.get(f"pred_n{i}")
            if fix_predictions:
                v = fix_prediction(protac, v, verbose=verbose)
            if v is None:
                continue
            score = score_split(
                protac, v,
                representative_e3s_fp=representative_e3s_fp,
                representative_whs_fp=representative_whs_fp,
            )
            n_flags, reasons = count_flags(score)
            key = _rank_key(score, n_flags, "Transformer")
            if best_key is None or key < best_key:
                best_key, best_pred, best_flags = key, v, (n_flags, reasons)
            if key[0] is False and key[1] == 0:
                break
        n_flags, reasons = best_flags
        return {
            protac_smiles_col: protac,
            "default_pred_n0": best_pred,
            "model_name": "Transformer",
            "heuristic_params": None,
            "n_flags": n_flags,
            "review_reasons": reasons,
        }

    return preds_ds.map(_map, desc="Transformer beam search (QC-scored)")


def _run_adaptive_ds(
    ds: Dataset,
    protac_smiles_col: str,
    grid: List[Tuple[float, bool]],
    representative_e3s_fp: List,
    representative_whs_fp: List,
    betweenness_approx_frac: Optional[float],
    use_xgboost: bool,
    xgboost_model: Optional[GraphEdgeClassifier],
    use_transformer: bool,
    pipe,
    batch_size: int,
    beam_size: int,
    fix_predictions: bool,
    num_proc: int,
    verbose: int,
) -> Dataset:
    """QC-gated escalation: heuristic parameter grid first (cheap, no model download),
    then XGBoost, then — only if explicitly enabled — the Transformer (GPU + beam
    search, by far the most expensive). Each stage only runs on rows the previous stage
    left flagged; a row is only replaced by a later stage's candidate when it scores
    strictly better under `score_split`, so escalation never regresses a row that
    already has a good split, and a structurally-valid-but-flagged candidate is always
    kept in preference to a stage that fails outright. Heuristic and XGBoost candidates
    are additionally checked with E3/POI roles swapped (`_best_orientation`) before
    they compete for a row, since their role assignment can be wrong even when the
    bond cuts themselves are right.

    Adds three columns beyond the usual [protac_smiles_col, "default_pred_n0",
    "model_name"] contract: "heuristic_params" (which grid point won, when
    model_name == "Heuristic"), and "n_flags" / "review_reasons" (from
    evaluation.count_flags) — so a batch run over test data also doubles as a QC pass,
    and the winning heuristic params across that set are a direct signal for what the
    package defaults should be.
    """
    from protac_splitter.evaluation import count_flags, score_split

    result_df = _run_heuristic_grid_ds(
        ds, protac_smiles_col, grid, representative_e3s_fp, representative_whs_fp,
        betweenness_approx_frac, num_proc, verbose,
    ).to_pandas()

    def _maybe_replace(row_i: int, protac: str, candidate_pred: Optional[str], model_name: str) -> None:
        current_score = score_split(
            protac, result_df.at[row_i, "default_pred_n0"],
            representative_e3s_fp=representative_e3s_fp,
            representative_whs_fp=representative_whs_fp,
        )
        current_n_flags, _ = count_flags(current_score)
        current_key = _rank_key(current_score, current_n_flags, result_df.at[row_i, "model_name"])

        candidate_score = score_split(
            protac, candidate_pred,
            representative_e3s_fp=representative_e3s_fp,
            representative_whs_fp=representative_whs_fp,
        )
        candidate_n_flags, candidate_reasons = count_flags(candidate_score)
        candidate_key = _rank_key(candidate_score, candidate_n_flags, model_name)

        if candidate_key < current_key:
            result_df.at[row_i, "default_pred_n0"] = candidate_pred
            result_df.at[row_i, "model_name"] = model_name
            result_df.at[row_i, "heuristic_params"] = None
            result_df.at[row_i, "n_flags"] = candidate_n_flags
            result_df.at[row_i, "review_reasons"] = candidate_reasons

    if use_xgboost:
        flagged = result_df.index[result_df["n_flags"] > 0].tolist()
        if flagged:
            xgb_ds = _run_xgboost_ds(
                ds.select(flagged), xgboost_model, representative_e3s_fp, protac_smiles_col,
                betweenness_threshold=grid[0][0], use_capacity_weight=grid[0][1],
                betweenness_approx_frac=betweenness_approx_frac,
            )
            for local_i, row_i in enumerate(flagged):
                protac = ds[row_i][protac_smiles_col]
                # XGBoost's own distinguish_fragments() role assignment can be wrong
                # even when the bond cuts are right (see _best_orientation) — check
                # both orientations before it competes with the current best.
                candidate_pred, _, _, _ = _best_orientation(
                    protac, xgb_ds[local_i]["default_pred_n0"],
                    representative_e3s_fp, representative_whs_fp, "XGBoost",
                )
                _maybe_replace(row_i, protac, candidate_pred, "XGBoost")

    if use_transformer:
        flagged = result_df.index[result_df["n_flags"] > 0].tolist()
        if flagged:
            transformer_ds = _run_transformer_scored_ds(
                ds.select(flagged), pipe, batch_size, beam_size, protac_smiles_col,
                fix_predictions, verbose, representative_e3s_fp, representative_whs_fp,
            )
            for local_i, row_i in enumerate(flagged):
                _maybe_replace(
                    row_i, ds[row_i][protac_smiles_col], transformer_ds[local_i]["default_pred_n0"], "Transformer",
                )

    return Dataset.from_pandas(result_df, preserve_index=False)


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
        adaptive_heuristic_grid: Optional[List[Tuple[float, bool]]] = None,
        adaptive_use_xgboost: bool = True,
        adaptive_use_transformer: bool = False,
        model: Optional[Literal[
            "transformer",
            "xgboost",
            "heuristic",
            "transformer->xgboost",
            "xgboost->heuristic",
            "xgboost+heuristic",
            "heuristic->xgboost",
            "heuristic+xgboost",
            "adaptive",
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
    | ``"adaptive"``                | QC-gated escalation: a small heuristic parameter      |
    |                               | grid first, then XGBoost, then (if enabled) the       |
    |                               | Transformer -- each stage only runs on molecules      |
    |                               | the previous stage left flagged by                    |
    |                               | evaluation.score_split. Slower than a single          |
    |                               | strategy but generally higher quality; also           |
    |                               | reports which method/params won. See the              |
    |                               | adaptive_* args below.                                |
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
        adaptive_heuristic_grid: Only used when ``model="adaptive"``. List of
            ``(betweenness_threshold, use_capacity_weight)`` pairs to try, in order,
            for the heuristic stage. ``None`` uses a built-in 6-point grid seeded with
            the package default first, so a molecule the default already splits
            cleanly costs exactly one heuristic call.
        adaptive_use_xgboost: Only used when ``model="adaptive"``. Whether the second
            stage (XGBoost) runs on molecules the heuristic grid left flagged.
            Default ``True``.
        adaptive_use_transformer: Only used when ``model="adaptive"``. Whether the
            third stage (Transformer) runs on molecules still flagged after XGBoost.
            Default ``False`` since it requires the ``[transformer]`` extra and a GPU
            is recommended; the first two stages already cover most cases.

    Returns:
        * Single string input → ``dict`` with keys ``protac_smiles_col``,
          ``"default_pred_n0"`` (``"e3.linker.poi"`` format), and ``"model_name"``.
        * List input → list of such dicts.
        * DataFrame input → DataFrame with the same columns.
        * ``model="adaptive"`` additionally includes ``"heuristic_params"`` (which
          grid point won, when ``model_name == "Heuristic"``, else ``None``),
          ``"n_flags"``, and ``"review_reasons"`` (from ``evaluation.count_flags``).
    """
    model_str = _resolve_model(use_transformer, use_xgboost, model)
    if model_str not in _VALID_MODELS:
        raise ValueError(
            f"`model` must be one of {sorted(_VALID_MODELS)}. Got: {model_str!r}"
        )

    # Load required resources up-front so errors surface before any SMILES processing.
    is_adaptive = model_str == "adaptive"
    needs_xgboost = "xgboost" in model_str or (is_adaptive and adaptive_use_xgboost)
    needs_transformer = "transformer" in model_str or (is_adaptive and adaptive_use_transformer)
    # Adaptive mode always needs both reference fingerprint sets to score candidates
    # (evaluation.score_split), regardless of whether XGBoost itself is enabled.
    needs_reference_fps = needs_xgboost or is_adaptive

    if needs_reference_fps:
        representative_e3s_fp = get_representative_e3s_fp()
    else:
        representative_e3s_fp = None
    representative_whs_fp = get_representative_whs_fp() if is_adaptive else None

    xgboost_model = load_graph_edge_classifier_from_cache() if needs_xgboost else None

    if needs_transformer:
        try:
            from protac_splitter.llms.model_utils import get_pipeline
        except ImportError:
            raise ImportError(
                "The 'transformer' model requires additional dependencies. "
                "Install them with:\n    pip install 'protac-splitter[transformer]'"
            ) from None
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
    elif model_str == "adaptive":
        preds_ds = _run_adaptive_ds(
            ds, protac_smiles_col,
            grid=adaptive_heuristic_grid or _DEFAULT_ADAPTIVE_GRID,
            representative_e3s_fp=representative_e3s_fp,
            representative_whs_fp=representative_whs_fp,
            betweenness_approx_frac=betweenness_approx_frac,
            use_xgboost=adaptive_use_xgboost,
            xgboost_model=xgboost_model,
            use_transformer=adaptive_use_transformer,
            pipe=pipe,
            batch_size=batch_size,
            beam_size=beam_size,
            fix_predictions=fix_predictions,
            num_proc=num_proc,
            verbose=verbose,
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
