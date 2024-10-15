from typing import Optional

from transformers import AutoTokenizer
import numpy as np
from rdkit import Chem, DataStructs
import evaluate

from ..evaluation import (
    # is_valid_smiles,
    # has_three_substructures,
    # has_all_attachment_points,
    # check_substructs,
    score_prediction,
)


def decode_and_get_metrics(
    pred,
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    rouge = None, # Optional[evaluate.metrics.rouge.Rouge] = None,
    fpgen = None, # Optional[Chem.rdFingerprintGenerator] = None,
) -> dict[str, float]:
    """ Compute metrics for tokenized PROTAC predictions.

    Args:
        pred (transformers.BatchEncoding): The predictions from the model.
        rouge (Rouge): The Rouge object to use for scoring. Example: `rouge = evaluate.load("rouge")`
        tokenizer (AutoTokenizer | str): The tokenizer to use for decoding the predictions. If a string, the tokenizer will be loaded using `AutoTokenizer.from_pretrained(tokenizer)`. Default: "seyonec/ChemBERTa-zinc-base-v1"
        fpgen (Chem.rdFingerprintGenerator): The fingerprint generator to use for computing the Tanimoto similarity. Default: `Chem.rdFingerprintGenerator.GetMorganGenerator(radius=8, fpSize=2048)`

    Returns:
        dict[str, float]: A dictionary containing the scores for the predictions
    """
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    input_ids = pred.inputs
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # Replace -100 in the IDs with the tokenizer pad token id
    # NOTE: Check the `ignore_index` argument in nn.CrossEntropyLoss.
    # TODO: Understand why this needs to be done to the inputs as well
    ignore_index = -100
    input_ids[input_ids == ignore_index] = tokenizer.pad_token_id
    labels_ids[labels_ids == ignore_index] = tokenizer.pad_token_id
    pred_ids[pred_ids == ignore_index] = tokenizer.pad_token_id
    
    # Get strings from IDs
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Get scores
    scores = []
    for pred_smiles, protac_smiles, label_smiles in zip(pred_str, input_str, label_str):
        scores.append(score_prediction(
            protac_smiles=protac_smiles,
            label_smiles=label_smiles,
            pred_smiles=pred_smiles,
            fpgen=fpgen,
            compute_rdkit_metrics=False,
            compute_graph_metrics=True,
            graph_edit_kwargs={"timeout": 0.5},
        ))
    scores = {k: np.array([s[k] for s in scores]).mean() for k in scores[0].keys()}
    
    # Get Rouge score
    if rouge is not None:
        rouge_output = rouge.compute(predictions=pred_str, references=label_str)
        scores.update({k: v for k, v in rouge_output.items()})

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

    return scores
