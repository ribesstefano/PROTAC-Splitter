import os
import requests
from typing import Union, Optional, Dict, List
from pathlib import Path
import logging

from datasets import Dataset
import pandas as pd

from protac_splitter.chemoinformatics import canonize
from protac_splitter.fixing_functions import fix_prediction
from protac_splitter.llms.model_utils import get_pipeline, run_pipeline
from protac_splitter.graphs.e3_clustering import get_representative_e3s_fp
from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier
from protac_splitter.graphs.splitting_algorithms import split_protac_graph_based


def load_graph_edge_classifier_from_cache(
    cache_dir: Union[str, Path] = "~/.cache/protac_splitter",
    model_filename: str = "PROTAC-Splitter-XGBoost.joblib",
    download_url: str = "https://docs.google.com/uc?export=download&id=1bb9i5_L_-re3QYPc7tSiCtVNEEbNIzAC",
) -> GraphEdgeClassifier:
    """
    Loads the GraphEdgeClassifier model from a local cache directory.
    If the model file is not found, downloads it from the specified URL.

    Args:
        cache_dir (str or Path): Directory to cache the model file.
        model_filename (str): Name of the model file.
        download_url (str): URL to download the model if not present.

    Returns:
        GraphEdgeClassifier: Loaded classifier.
    """
    cache_dir = Path(os.path.expanduser(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / model_filename

    if not model_path.exists():
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        expected_size = int(response.headers.get("Content-Length", -1))

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

        if expected_size != -1:
            actual = model_path.stat().st_size
            if actual != expected_size:
                raise RuntimeError(f"Download incomplete: got {actual}, expected {expected_size}")

        # Optional checksum:
        # NOTE: Uncomment the following for debugging
        import hashlib
        h = hashlib.sha256(model_path.read_bytes()).hexdigest()
        h_orig = "513621f4dc2ff7ec819a222bc7311afb8b6e6e89d6d694dd2906e695a50086dd"
        if h != h_orig:
            raise RuntimeError(
                f"Downloaded model checksum mismatch: got {h}, expected {h_orig}. "
                "Please delete the model file and try again."
            )

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
            model_name="ailab-bio/PROTAC-Splitter-EncoderDecoder-lr_reduce-rand-smiles",
            token=os.environ.get("HF_TOKEN", None),
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

    # if tokenizer is None:
    #     if verbose:
    #         print(f"Loading tokenizer...")
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # if pipe is None:
    #     if verbose:
    #         print("Loading pipeline for \"default\" predictions...")
    #     pipe = pipeline(
    #         "text2text-generation",
    #         model=model_name,
    #         tokenizer=tokenizer,
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         token=hf_token,
    #         num_return_sequences=beam_size,
    #     )

    # if isinstance(protac_smiles, str):
    #     protac_smiles_canon = canonize(protac_smiles)
    #     if protac_smiles_canon is None:
    #         raise ValueError(f"Invalid PROTAC SMILES: {protac_smiles}")
    #     pred = pipe(protac_smiles_canon)
    #     pred = {f"default_pred_n{i}": pred[i]["generated_text"] for i in range(len(pred))}
    #     if fix_predictions:
    #         p_fixed = {k: fix_prediction(protac_smiles_canon, v, verbose=verbose) for k, v in pred.items()}
    #         # For each prediction, if the fixed prediction is not None, we
    #         # replace the original prediction with the fixed one.
    #         for k, v in p_fixed.items():
    #             if v is not None:
    #                 pred[k] = v
    #     preds = [pred]

    # if isinstance(protac_smiles, list):
    #     # Canonize and check if all PROTAC SMILES are valid
    #     protac_smiles_canon = [canonize(protac) for protac in protac_smiles]
    #     if None in protac_smiles_canon:
    #         wrong_protacs = [protac for protac, canon in zip(protac_smiles, protac_smiles_canon) if canon is None]
    #         raise ValueError(f"Invalid PROTAC SMILES in list: {wrong_protacs}")

    #     # Get the predictions for all PROTAC SMILES
    #     preds = pipe(protac_smiles_canon, batch_size=batch_size)
    #     preds = [{f"default_pred_n{i}": p["generated_text"] for i, p in enumerate(pred)} for pred in preds]

    #     if fix_predictions:
    #         for i, (protac, pred) in enumerate(zip(protac_smiles_canon, preds)):
    #             p_fixed = {k: fix_prediction(protac, v, verbose=verbose) for k, v in pred.items()}
    #             # For each prediction, if the fixed prediction is not None, we
    #             # replace the original prediction with the fixed one.
    #             for k, v in p_fixed.items():
    #                 if v is not None:
    #                     preds[i][k] = v

    # if isinstance(protac_smiles, pd.DataFrame):
    #     # Check if the DataFrame contains a columns named `protac_smiles_col`
    #     if protac_smiles_col not in protac_smiles.columns:
    #         raise ValueError(f"DataFrame must contain a column named \"{protac_smiles_col}\".")
        
    #     # Canonize and check if all PROTAC SMILES are valid
    #     protac_smiles_canon = protac_smiles.apply(lambda x: canonize(x[protac_smiles_col]), axis=1)

    #     # Check if there are invalid PROTAC SMILES
    #     if protac_smiles_canon.isnull().any():
    #         wrong_protacs = protac_smiles[protac_smiles_canon.isnull()]
    #         raise ValueError(f"Invalid PROTAC SMILES in DataFrame: {wrong_protacs}")

    #     # Convert the Series to a DataFrame
    #     protac_smiles_canon = pd.DataFrame(protac_smiles_canon, columns=[protac_smiles_col])
        
    #     # Convert the DataFrame to a Dataset
    #     dataset = Dataset.from_pandas(protac_smiles_canon)
    #     preds = []
    #     for pred in tqdm(pipe(KeyDataset(dataset, protac_smiles_col), batch_size=batch_size), total=len(dataset) // batch_size, desc="Generating predictions"):
    #         p = {f"default_pred_n{i}": pred[i]["generated_text"] for i in range(len(pred))}
    #         preds.append(p)

    #     if fix_predictions:
    #         for i, (protac, pred) in tqdm(enumerate(zip(protac_smiles_canon, preds)), desc="Fixing predictions", total=len(preds)):
    #             p_fixed = {k: fix_prediction(protac, v, verbose=verbose) for k, v in pred.items()}
    #             # For each prediction, if the fixed prediction is not None, we
    #             # replace the original prediction with the fixed one.
    #             for k, v in p_fixed.items():
    #                 if v is not None:
    #                     pred[k] = v

    # if return_check_reassembly:
    #     if isinstance(protac_smiles_canon, str):
    #         protac_smiles_list = [protac_smiles_canon]
    #     elif isinstance(protac_smiles_canon, list):
    #         protac_smiles_list = protac_smiles_canon
    #     elif isinstance(protac_smiles_canon, pd.DataFrame):
    #         protac_smiles_list = protac_smiles_canon[protac_smiles_col].tolist()
        
    #     print("Checking re-assembly...")
    #     for protac, pred in zip(protac_smiles_list, preds):
    #         for i in range(beam_size):
    #             pred[f"reassembly_correct_n{i}"] = check_reassembly(protac, pred[f"default_pred_n{i}"])

    #     # Just take the first prediction if the input was a string
    #     if isinstance(protac_smiles, str):
    #         preds = preds[0]

    # return preds