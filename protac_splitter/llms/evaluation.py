from transformers import AutoTokenizer
import numpy as np
from rdkit import Chem, DataStructs
import evaluate

from ..evaluation import (
    is_valid_smiles,
    has_three_substructures,
    has_all_attachment_points,
    check_substructs,
)


def split_prediction(
        pred: str,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> dict[str, str] | None:
    """ Split a PROTAC SMILES prediction into its three substructures.

    Args:
        pred (str): The SMILES notation for the PROTAC molecule.
        poi_attachment_id (int): The attachment point ID for the POI substructure.
        e3_attachment_id (int): The attachment point ID for the E3 substructure.

    Returns:
        dict[str, str] | None: A dictionary containing the SMILES notations for the POI, linker, and E3 substructures, or None if the prediction is invalid
    """
    sbstructs = pred.split('.')
    if len(sbstructs) != 3:
        return None
    ret = {}
    for substr in sbstructs:
        if f'[*:{poi_attachment_id}]' in substr and f'[*:{e3_attachment_id}]' not in substr:
            ret['poi'] = substr
        elif f'[*:{e3_attachment_id}]' in substr and f'[*:{poi_attachment_id}]' not in substr:
            ret['e3'] = substr
        elif f'[*:{poi_attachment_id}]' in substr and f'[*:{e3_attachment_id}]' in substr:
            ret['linker'] = substr
        else:
            return None
    return ret


def compute_metrics_with_chem(
    pred,
    rouge = evaluate.load("rouge"),
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=8, fpSize=2048),
):
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    input_ids = pred.inputs
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    # Replace -100 in the IDs with the tokenizer pad token id
    # NOTE: Check the `ignore_index` argument in nn.CrossEntropyLoss.
    ignore_index = -100
    labels_ids[labels_ids == ignore_index] = tokenizer.pad_token_id
    # Get strings from IDs
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # Get Rouge score
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    scores = {k: round(v, 4) for k, v in rouge_output.items()}
    # Get valid SMILES score
    valid_smiles = np.array([is_valid_smiles(s) for s in pred_str])
    scores['valid_smiles'] = valid_smiles.astype(int).mean()
    # Get has_three_substructures score
    num_substructures = np.array([has_three_substructures(s) for s in pred_str])
    scores['has_three_substructures'] = num_substructures.astype(int).mean()
    # Get has_all_attachment_points score
    num_attach_points = np.array([has_all_attachment_points(s) for s in pred_str])
    scores['has_all_attachment_points'] = num_attach_points.astype(int).mean()

    print('=' * 80)
    print(pred)
    print(pred.inputs)
    print(pred.predictions)
    print(pred_str)
    print(label_str)
    print('=' * 80)

    # Check if re-combining the substructures results in the original PROTAC
    checks = []
    for i, (pred_smiles, protac_smiles, label_smiles) in enumerate(zip(pred_str, input_str, label_str)):
        if i < 5:
            print(f'protac: {protac_smiles}')
            print(f'label:  {label_smiles}')
            print(f'pred:   {pred_smiles}')
            print(f'\t- valid: {is_valid_smiles(pred_smiles)}')
            print(f'\t- has_three_substructures: {has_three_substructures(pred_smiles)}')
            print(f'\t- has_all_attachment_points: {has_all_attachment_points(pred_smiles)}')
            for j, s in enumerate(pred_smiles.split('.')):
                print(f'Substruct n.{j}: {s} (valid: {is_valid_smiles(s)})')
            print('-' * 80)
        substructs = split_prediction(pred_smiles)
        if substructs is None:
            checks.append(False)
            continue
        checks.append(check_substructs(
            protac_smiles,
            substructs['poi'],
            substructs['linker'],
            substructs['e3'],
        ))
    scores['reassembly'] = np.array(checks).astype(int).mean()

    # Count how many times the character '*' appears in the prediction
    num_stars = np.array([s.count('*') for s in pred_str])
    scores['num_stars'] = num_stars.mean()

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