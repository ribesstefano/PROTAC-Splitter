import hashlib
import logging
import requests
from pathlib import Path
from typing import Union, Optional, Dict, List

from datasets import Dataset
import pandas as pd

from protac_splitter.config import get_cache_dir, get_hf_token
from protac_splitter.chemoinformatics import canonize
from protac_splitter.fixing_functions import fix_prediction
from protac_splitter.llms.model_utils import get_pipeline, run_pipeline
from protac_splitter.graphs.e3_clustering import get_representative_e3s_fp
from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier
from protac_splitter.graphs.splitting_algorithms import split_protac_graph_based

_XGBOOST_MODEL_FILENAME = "PROTAC-Splitter-XGBoost.joblib"
_XGBOOST_DOWNLOAD_URL = (
    "https://zenodo.org/records/15797310/files/"
    "PROTAC-Splitter-XGBoost.joblib?download=1"
)
_XGBOOST_SHA256 = "513621f4dc2ff7ec819a222bc7311afb8b6e6e89d6d694dd2906e695a50086dd"


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


def split_protac(
        protac_smiles: Union[str, List, pd.DataFrame],
        use_transformer: bool = False,
        use_xgboost: bool = True,
        fix_predictions: bool = True,
        protac_smiles_col: str = "text",
        batch_size: int = 1,
        beam_size: int = 5,
        device: Optional[Union[int, str]] = None,
        num_proc: int = 1,
        verbose: int = 0,
        betweenness_threshold: float = 0.4,
        use_capacity_weight: bool = False,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """ Split a PROTAC SMILES into the two ligands and the linker.

    If `use_transformer` and `use_xgboost` are both True, the Transformer model
    will run first, and XGBost will be used as a fallback for predictions that
    fail re-assembly and fixing. If both `use_transformer` and `use_xgboost`
    are False, a fully heuristic-based algorithm will be used for splitting.

    Args:
        protac_smiles (str, list, or pd.DataFrame): The PROTAC SMILES to split.
            If a DataFrame is provided, it must contain a column named `protac_smiles_col`.
        use_transformer (bool): Whether to use the transformer model for splitting.
        use_xgboost (bool): Whether to use the XGBoost model for splitting.
        fix_predictions (bool): Whether to fix the predictions using deterministic cheminformatics rules. Only used if `use_transformer` is True.
        protac_smiles_col (str): The name of the column containing the PROTAC SMILES in the DataFrame.
        batch_size (int): Batch size for processing. Only used if `use_transformer` is True.
        beam_size (int): Number of beam search predictions to generate. Only used if `use_transformer` is True. Higher values may yield better results but increase computation time.
        device (int or str, optional): Device to run the Transformer model on. Defaults to None will attempt to run on GPU if available, otherwise CPU.
        num_proc (int): Number of processes to use for parallel processing. Useful for large datasets of PROTACs to split.
        verbose (int): Verbosity level.
        betweenness_threshold (float): Betweenness-centrality threshold used by the heuristic algorithm to identify split points. Higher values are more conservative (fewer cuts). Default 0.4.
        use_capacity_weight (bool): Whether to weight edges by bond capacity when computing betweenness centrality (heuristic algorithm only). Default False.
    
    Returns:
        Union[Dict[str, str], List[Dict[str, str]]]: Depending on the input type, returns:
            - If a single string is provided, returns a dictionary with format: `{protac_smiles_col: protac_smiles, "default_pred_n0": e3l.linker.warhead, "model_name": Transformer|XGBoost|Heuristic}`.
            - If a list of strings is provided, returns a list of dictionaries with the same format as above.
            - If a DataFrame is provided, returns a DataFrame with columns: `protac_smiles_col`, `default_pred_n0`, and `model_name`. The `default_pred_n0` column contains the predicted split strings in the format `e3.linker.warhead`.
    """
    if use_xgboost:
        representative_e3s_fp = get_representative_e3s_fp()
        xgboost_model = load_graph_edge_classifier_from_cache()
        
    # Generate a Dataset from the input PROTAC SMILES
    if isinstance(protac_smiles, str):
        protac_smiles_canon = canonize(protac_smiles)
        if protac_smiles_canon is None:
            raise ValueError(f"Invalid PROTAC SMILES: {protac_smiles}")
        ds = Dataset.from_dict({protac_smiles_col: [protac_smiles_canon]})
    elif isinstance(protac_smiles, list):
        # Canonize and check if all PROTAC SMILES are valid
        protac_smiles_canon = [canonize(protac) for protac in protac_smiles]
        if None in protac_smiles_canon:
            wrong_protacs = [protac for protac, canon in zip(protac_smiles, protac_smiles_canon) if canon is None]
            raise ValueError(f"Invalid PROTAC SMILES in list: {wrong_protacs}")
        ds = Dataset.from_dict({protac_smiles_col: protac_smiles_canon})
    elif isinstance(protac_smiles, pd.DataFrame):
        # Check if the DataFrame contains a columns named `protac_smiles_col`
        if protac_smiles_col not in protac_smiles.columns:
            raise ValueError(f"DataFrame must contain a column named \"{protac_smiles_col}\".")
        # Canonize and check if all PROTAC SMILES are valid
        protac_smiles_canon = protac_smiles[protac_smiles_col].apply(canonize)
        if protac_smiles_canon.isnull().any():
            wrong_protacs = protac_smiles[protac_smiles_canon.isnull()]
            raise ValueError(f"Invalid PROTAC SMILES in DataFrame: {wrong_protacs}")
        ds = Dataset.from_pandas(protac_smiles_canon.to_frame(name=protac_smiles_col))
    
    if use_transformer:
        pipe = get_pipeline(
            model_name="ailab-bio/PROTAC-Splitter",
            token=get_hf_token(),
            is_causal_language_model=False,
            num_return_sequences=beam_size,
            device=device,
        )

        # preds will be a list of dictionaries, each containing the
        # beam-size predictions for each input PROTAC SMILES. Format: [{'pred_n0': 'prediction_0', 'pred_n1': 'prediction_1', ...}, ...]
        preds = run_pipeline(
            pipe,
            ds,
            batch_size,
            is_causal_language_model=False,
            smiles_column=protac_smiles_col,
        )

        # Turn the predictions into a DataFrame and then into a Dataset
        preds_df = pd.DataFrame(preds)
        preds_df[protac_smiles_col] = ds[protac_smiles_col]
        preds_ds = Dataset.from_pandas(preds_df)

        def mapping_func(row: Dict[str, str]) -> Dict[str, str]:
            """Fix the predictions for each row."""
            protac = row[protac_smiles_col]
            if fix_predictions:
                preds = {k: fix_prediction(protac, v, verbose=verbose) for k, v in row.items() if k.startswith("pred_")}
            else:
                preds = {k: v for k, v in row.items() if k.startswith("pred_")}

            # If all preds are None, we attempt to use the XGBoost model
            if all(v is None for v in preds.values()):
                if use_xgboost:
                    pred = split_protac_graph_based(
                        protac_smiles=protac,
                        use_classifier=True,
                        classifier=xgboost_model,
                        representative_e3s_fp=representative_e3s_fp,
                        betweenness_threshold=betweenness_threshold,
                        use_capacity_weight=use_capacity_weight,
                    )
                    return {
                        protac_smiles_col: protac,
                        "default_pred_n0": f"{pred['e3']}.{pred['linker']}.{pred['poi']}",
                        "model_name": "XGBoost",
                    }
                else:
                    # If no predictions are valid, we return None for the default prediction
                    return {
                        protac_smiles_col: protac,
                        "default_pred_n0": None,
                        "model_name": "Transformer",
                    }
            else:
                # Select the non-None prediction with the lowest beam index
                # NOTE: The HF predictions comes in lists, with the first
                # element being the one with the highest likelihood.
                for i in range(beam_size):
                    key = f"pred_n{i}"
                    if preds[key] is not None:
                        return {
                            protac_smiles_col: protac,
                            "default_pred_n0": preds[key],
                            "model_name": "Transformer",
                        }

        # Map the function over the Dataset to fix the predictions and/or
        # replace them with the XGBoost fallback predictions if they fail.
        if fix_predictions or use_xgboost:
            preds_ds = preds_ds.map(
                mapping_func,
                num_proc=1 if use_xgboost else num_proc, # Using XGBoost IN a map function might not be thread-safe
                desc=f"{'Fixing predictions' if fix_predictions else ''}{' and ' if fix_predictions and use_xgboost else ''}{'Replacing predictions with XGBoost fallback' if use_xgboost else ''}",
            )

    elif use_xgboost:
        # Use the XGBoost model only
        def mapping_func(row: Dict[str, str]) -> Dict[str, str]:
            """Split the PROTAC SMILES using the XGBoost model."""
            protac = row[protac_smiles_col]
            pred = split_protac_graph_based(
                protac_smiles=protac,
                use_classifier=True,
                classifier=xgboost_model,
                representative_e3s_fp=representative_e3s_fp,
                betweenness_threshold=betweenness_threshold,
                use_capacity_weight=use_capacity_weight,
            )
            if all(v is None for v in pred.values()):
                split = None
            else:
                split = f"{pred['e3']}.{pred['linker']}.{pred['poi']}"
            return {
                protac_smiles_col: protac,
                "default_pred_n0": split,
                "model_name": "XGBoost",
            }
        preds_ds = ds.map(
            mapping_func,
            num_proc=1,
            desc="Splitting PROTAC SMILES using XGBoost model",
        )
    else:
        # If neither transformer nor XGBoost is used, we use the heuristic-based
        # algorithm, that does not require any model.
        def mapping_func(row: Dict[str, str]) -> Dict[str, str]:
            """Split the PROTAC SMILES using the heuristic-based algorithm."""
            protac = row[protac_smiles_col]
            pred = split_protac_graph_based(
                protac_smiles=protac,
                use_classifier=False,
                betweenness_threshold=betweenness_threshold,
                use_capacity_weight=use_capacity_weight,
            )
            if all(v is None for v in pred.values()):
                split = None
            else:
                split = f"{pred['e3']}.{pred['linker']}.{pred['poi']}"
            return {
                protac_smiles_col: protac,
                "default_pred_n0": split,
                "model_name": "Heuristic",
            }
        preds_ds = ds.map(
            mapping_func,
            num_proc=num_proc,
            desc="Splitting PROTAC SMILES using heuristic-based algorithm",
        )

    if isinstance(protac_smiles, str):
        # If the input was a single string, we return the first prediction
        return preds_ds[0]
    elif isinstance(protac_smiles, pd.DataFrame):
        # If the input was a DataFrame, we return a dataframe with the predictions
        return preds_ds.to_pandas()
    elif isinstance(protac_smiles, list):
        # Convert the Dataset to a list of dictionaries
        return [row for row in preds_ds]