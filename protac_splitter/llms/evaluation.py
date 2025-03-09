from typing import Union

from transformers import AutoTokenizer, EvalPrediction
import numpy as np
from rdkit import Chem, DataStructs
import evaluate
import multiprocessing as mp
import datetime

from protac_splitter.evaluation import (
    # is_valid_smiles,
    # has_three_substructures,
    # has_all_attachment_points,
    # check_substructs,
    score_prediction,
)

def process_predictions(args) -> list:
    """ Process one iteration of the prediction scoring.
    
    Args:
        args (tuple): Tuple of arguments for the scoring function.

    Returns:
        dict: The scores for the prediction.
    """
    pred_smiles, protac_smiles, label_smiles, fpgen, compute_rdkit_metrics, compute_graph_metrics = args
    scores = []
    for protac, pred, label in zip(protac_smiles, pred_smiles, label_smiles):
        scores.append(score_prediction(
            protac_smiles=protac,
            label_smiles=label,
            pred_smiles=pred,
            fpgen=fpgen,
            compute_rdkit_metrics=compute_rdkit_metrics,
            compute_graph_metrics=compute_graph_metrics,
            graph_edit_kwargs={"timeout": 0.05},
        ))
    return scores


def decode_and_get_metrics(
        pred: EvalPrediction,
        tokenizer: Union[AutoTokenizer, str] = "seyonec/ChemBERTa-zinc-base-v1",
        rouge = None, # Optional[evaluate.metrics.rouge.Rouge] = None,
        fpgen = None, # Optional[Chem.rdFingerprintGenerator] = None,
        compute_rdkit_metrics: bool = False,
        compute_graph_metrics: bool = True,
        num_proc: int = 1,
        batch_size: int = 128,
        use_nan_for_missing: bool = True,
        causal_language_modeling: bool = False,
) -> dict[str, float]:
    """ Compute metrics for tokenized PROTAC predictions.

    Args:
        pred (transformers.EvalPrediction): The predictions from the model.
        rouge (Rouge): The Rouge object to use for scoring. Example: `rouge = evaluate.load("rouge")`
        tokenizer (AutoTokenizer | str): The tokenizer to use for decoding the predictions. If a string, the tokenizer will be loaded using `AutoTokenizer.from_pretrained(tokenizer)`. Default: "seyonec/ChemBERTa-zinc-base-v1"
        fpgen (Chem.rdFingerprintGenerator): The fingerprint generator to use for computing the Tanimoto similarity. Default: `Chem.rdFingerprintGenerator.GetMorganGenerator(radius=8, fpSize=2048)`

    Returns:
        dict[str, float]: A dictionary containing the scores for the predictions
    """
    print(f"[{datetime.datetime.now()}] Starting decode_and_get_metrics (protac_splitter/llms/evaluation.py)")

    if causal_language_modeling:
        # NOTE: For causal language models, we only care about perplexity, so we
        # only need the eval_loss, which is automatically added.
        return {}

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    input_ids = pred.inputs
    
    if causal_language_modeling:
        # The prediction logits will be of shape: (batch_size, sequence_length, vocabulary_size)
        # So we need to get the argmax of the last dimension to get the
        # predicted token IDs.
        # NOTE: Not exactly the same as what would happen during generation, but
        # hopefully it's close enough to assess model performance during
        # training.
        pred_ids = np.argmax(pred_ids, axis=-1)
    
    # Replace -100 in the IDs with the tokenizer pad token id
    # NOTE: Check the `ignore_index` argument in nn.CrossEntropyLoss.
    # TODO: Understand why this needs to be done to the inputs as well
    ignore_index = -100
    labels_ids[labels_ids == ignore_index] = tokenizer.pad_token_id
    pred_ids[pred_ids == ignore_index] = tokenizer.pad_token_id
    
    # Get strings from IDs
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    if not causal_language_modeling:
        input_ids[input_ids == ignore_index] = tokenizer.pad_token_id
        input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    else:
        # NOTE: For causal language models, i.e., decoder only, the input PROTAC
        # is in the label. Therefore, we need to decode the label to get the
        # input. The label looks something like "PROTAC.E3.Linker.WH", so we
        # need to split it and get the last (three) parts.
        input_str = [str(s.split('.')[0]) for s in label_str]
        label_str = ['.'.join(s.split('.')[1:]) for s in label_str]
        pred_str = ['.'.join(s.split('.')[1:]) if '.' in s else s for s in pred_str]

    # Get scores
    if num_proc == 1:
        scores = process_predictions((
            pred_str, input_str, label_str, fpgen, compute_rdkit_metrics, compute_graph_metrics
        ))
    else:
        # Use pools to process batches of predictions
        with mp.Pool(processes=num_proc) as pool:
            scores = []
            for i in range(0, len(pred_str), batch_size):
                scores += pool.map(process_predictions, [
                    (pred_str[i:i+batch_size], input_str[i:i+batch_size], label_str[i:i+batch_size], fpgen, compute_rdkit_metrics, compute_graph_metrics)
                ])
            # Flatten the list of scores
            scores = [s for ls in scores for s in ls]

    # Aggregate scores
    scores_labels = set()
    for s in scores:
        scores_labels.update(s.keys())

    aggregated_scores = {}
    for k in scores_labels:
        values = np.array([s.get(k, np.nan) for s in scores], dtype=float)

        # If values is all NaN, set the aggregated score to NaN and continue
        if np.all(np.isnan(values)):
            aggregated_scores[k] = None
            continue

        # Compute average, excluding `NaN` values if necessary
        if use_nan_for_missing:
            aggregated_scores[k] = np.nanmean(values)
        else:
            valid_values = values[~np.isnan(values)]
            aggregated_scores[k] = np.mean(valid_values) if valid_values.size > 0 else float('nan')

    # Get Rouge score
    if rouge is not None:
        rouge_output = rouge.compute(predictions=pred_str, references=label_str)
        aggregated_scores.update({k: v for k, v in rouge_output.items()})

    # TODO
    # # Get tanimoto score
    # pred_str = np.array(pred_str)[valid_smiles == 1]
    # label_str = np.array(label_str)[valid_smiles == 1]
    # if len(pred_str) == 0:
    #     scores['tanimoto'] = 0.0
    #     return scores
    # pred_mols = [Chem.MolFromSmiles(s) for s in pred_str]
    # label_mols = [Chem.MolFromSmiles(s) for s in label_str]
    # pred_fps = [fpgen.GetFingerprint(m) for m in pred_mols]
    # label_fps = [fpgen.GetFingerprint(m) for m in label_mols]
    # tanimoto = [DataStructs.TanimotoSimilarity(l, p) for l, p in zip(label_fps, pred_fps)]
    # scores['tanimoto'] = np.array(tanimoto).mean()

    print(f"[{datetime.datetime.now()}] Done with decode_and_get_metrics (protac_splitter/llms/evaluation.py)")

    return aggregated_scores
