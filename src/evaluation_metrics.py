import evaluate
import numpy as np
from transformers import AutoTokenizer
from typing import Optional
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

def compute_metrics(
    pred,
    rouge = evaluate.load("rouge"),
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
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
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
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


def is_substructure(protac_smiles, substruct_smiles) -> bool:
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    substruct_mol = Chem.MolFromSmarts(substruct_smiles)
    return protac_mol.HasSubstructMatch(substruct_mol)


def same_atom_counts_and_types(smiles1, smiles2, get_atoms_diff=False):
    """
    Check if two molecules have the same number and types of atoms.

    Args:
    smiles1 (str): SMILES notation for the first molecule.
    smiles2 (str): SMILES notation for the second molecule.

    Returns:
    bool: True if the molecules have the same atom counts and types, False otherwise.
    """
    if isinstance(smiles1, str):
        mol1 = Chem.MolFromSmiles(smiles1)
    else:
        mol1 = smiles1
    if isinstance(smiles2, str):
        mol2 = Chem.MolFromSmiles(smiles2)
    else:
        mol2 = smiles2
    if mol1 is None or mol2 is None:
        if get_atoms_diff:
            return False
            # raise ValueError("Invalid SMILES notation provided for one or both molecules.")
        else:
            return False
    num_atoms1 = Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol1)
    num_atoms2 = Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol2)
    if get_atoms_diff:
        return abs(num_atoms1 - num_atoms2)
        # tmp = {}
        # for atom in atom_counts1.keys():
        #     tmp[atom] = int(abs(atom_counts1.get(atom, 0) - atom_counts2.get(atom, 0)))
        # for atom in atom_counts2.keys():
        #     tmp[atom] = int(abs(atom_counts1.get(atom, 0) - atom_counts2.get(atom, 0)))
        # return tmp # abs(atom_counts1.get('O', 0) - atom_counts2.get('O', 0))
    else:
        atom_counts1, atom_counts2 = {}, {}
        for atom in mol1.GetAtoms():
            if '*' not in atom.GetSmarts():
                atom_counts1[atom.GetSymbol()] = atom_counts1.get(atom.GetSymbol(), 0) + 1
        for atom in mol2.GetAtoms():
            if '*' not in atom.GetSmarts():
                atom_counts2[atom.GetSymbol()] = atom_counts2.get(atom.GetSymbol(), 0) + 1
        return (atom_counts1 == atom_counts2) & (num_atoms1 == num_atoms2)


def reward_function(query, response) -> float:
    if not has_three_substructures(response):
        return 0.
    if not has_all_attachment_points(response):
        return 0.
    response_mol = Chem.MolFromSmiles(response)
    if response_mol is None:
        return 0.
    if not same_atom_counts_and_types(response_mol, query):
        return 0.
    # return 1. - same_atom_counts_and_types(response, query, get_atoms_diff=True)
    # substructures = response.split(".")
    # for substructure in substructures:
    #     if not is_substructure(query, response):
    #         return 0.
    return 1.