import evaluate
import numpy as np
from transformers import AutoTokenizer
from typing import Optional
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

def compute_metrics(
    pred,
    rouge = evaluate.load("rouge"),
    tokenizer: [AutoTokenizer, str] = "seyonec/ChemBERTa-zinc-base-v1",
):
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    return {k: round(v, 4) for k, v in rouge_output.items()}


def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return 1 if mol is not None else 0


def has_three_substructures(smiles: str) -> bool:
    return smiles.count(".") == 2


def has_all_attachment_points(smiles: str) -> bool:
    return smiles.count("[*:1]") == 2 and smiles.count("[*:2]") == 2


def compute_metrics_with_chem(
    pred,
    rouge = evaluate.load("rouge"),
    tokenizer: [AutoTokenizer, str] = "seyonec/ChemBERTa-zinc-base-v1",
    fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=8, fpSize=2048),
):
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    # Replace -100 in the IDs with the tokenizer pad token id
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    # Get strings from IDs
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # Get Rouge score
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    scores = {k: round(v, 4) for k, v in rouge_output.items()}
    # Get valid SMILES score
    valid_smiles = np.array([is_valid_smiles(s) for s in pred_str])
    scores['valid_smiles'] = valid_smiles.mean()
    # Get has_three_substructures score
    num_substructures = np.array([has_three_substructures(s) for s in pred_str])
    scores['has_three_substructures'] = num_substructures.mean()
    # Get has_all_attachment_points score
    num_attach_points = np.array([has_all_attachment_points(s) for s in pred_str])
    scores['has_all_attachment_points'] = num_attach_points.mean()
    # Get tanimoto score
    pred_str = np.array(pred_str)[valid_smiles == 1]
    label_str = np.array(label_str)[valid_smiles == 1]
    if len(pred_str) == 0:
        scores['tanimoto'] = 0.0
        return scores
    pred_mols = [Chem.MolFromSmiles(s) for s in pred_str]
    label_mols = [Chem.MolFromSmiles(s) for s in label_str]
    pred_fps = [fpgen.GetFingerprint(m) for m in pred_mols]
    label_fps = [fpgen.GetFingerprint(m) for m in label_mols]
    tanimoto = [DataStructs.TanimotoSimilarity(l, p) for l, p in zip(label_fps, pred_fps)]
    scores['tanimoto'] = np.array(tanimoto).mean()
    return scores