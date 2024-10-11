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
    rouge = evaluate.load("rouge"),
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=11, fpSize=2048),
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
    
    # Get Rouge score
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)

    # Get other scores
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
    scores.update({k: v for k, v in rouge_output.items()})

    return scores

    # scores = {k: v for k, v in rouge_output.items()}
    # # Get valid SMILES score
    # valid_smiles = np.array([is_valid_smiles(s) for s in pred_str])
    # scores['valid_smiles'] = valid_smiles.astype(int).mean()
    # # Get has_three_substructures score
    # num_substructures = np.array([has_three_substructures(s) for s in pred_str])
    # scores['has_three_substructures'] = num_substructures.astype(int).mean()
    # # Get has_all_attachment_points score
    # num_attach_points = np.array([has_all_attachment_points(s) for s in pred_str])
    # scores['has_all_attachment_points'] = num_attach_points.astype(int).mean()

    # # Check if re-combining the substructures results in the original PROTAC
    # checks = []
    # for i, (pred_smiles, protac_smiles, label_smiles) in enumerate(zip(pred_str, input_str, label_str)):
    #     if i < 1:
    #         print(f'protac: {protac_smiles}')
    #         print(f'label:  {label_smiles}')
    #         print(f'pred:   {pred_smiles}')
    #         print(f'\t- valid: {is_valid_smiles(pred_smiles)}')
    #         print(f'\t- has_three_substructures: {has_three_substructures(pred_smiles)}')
    #         print(f'\t- has_all_attachment_points: {has_all_attachment_points(pred_smiles)}')
    #         for j, s in enumerate(pred_smiles.split('.')):
    #             print(f'Substruct n.{j}: {s} (valid: {is_valid_smiles(s)})')
    #         print('-' * 80)
    #     substructs = split_prediction(pred_smiles)
    #     if any(v is None for v in substructs.values()):
    #         checks.append(False)
    #         continue
    #     checks.append(check_substructs(
    #         protac_smiles,
    #         substructs['poi'],
    #         substructs['linker'],
    #         substructs['e3'],
    #     ))
    # scores['reassembly'] = np.array(checks).astype(int).mean()

    # # # Count how many times the character '*' appears in the prediction
    # # num_stars = np.array([s.count('*') for s in pred_str])
    # # scores['num_stars'] = num_stars.mean()

    # # # Get tanimoto score
    # # pred_str = np.array(pred_str)[valid_smiles == 1]
    # # label_str = np.array(label_str)[valid_smiles == 1]
    # # if len(pred_str) == 0:
    # #     scores['tanimoto'] = 0.0
    # #     return scores
    # # pred_mols = [Chem.MolFromSmiles(s) for s in pred_str]
    # # label_mols = [Chem.MolFromSmiles(s) for s in label_str]
    # # pred_fps = [fpgen.GetFingerprint(m) for m in pred_mols]
    # # label_fps = [fpgen.GetFingerprint(m) for m in label_mols]
    # # tanimoto = [DataStructs.TanimotoSimilarity(l, p) for l, p in zip(label_fps, pred_fps)]
    # # scores['tanimoto'] = np.array(tanimoto).mean()
    # return scores

