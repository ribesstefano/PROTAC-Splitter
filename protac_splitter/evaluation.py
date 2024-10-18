""" Evaluation functions for the protac_splitter package. They need to be generic to accomodate predictions coming from different models. """

import logging
from typing import Tuple, Any, Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

# Disable RDKit logging: when checking SMILES validity, we suppress warnings
RDLogger.DisableLog("rdApp.*")

from .chemoinformatics import canonize_smiles, remove_stereo
from .protac_cheminformatics import reassemble_protac
from .graphs_utils import (
    get_smiles2graph_edit_distance,
    get_smiles2graph_edit_distance_norm,
)


def is_valid_smiles(smiles: str, return_mol: bool = False) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if return_mol:
        return mol is not None, mol
    return mol is not None


def has_three_substructures(smiles: str) -> bool:
    return smiles.count(".") == 2


def has_all_attachment_points(smiles: str) -> bool:
    return smiles.count("[*:1]") == 2 and smiles.count("[*:2]") == 2


def is_substructure(protac_smiles: str, substruct_smiles: str) -> bool:
    """ Check if a molecule is a substructure of another molecule.

    Args:
        protac_smiles (str): The SMILES of the PROTAC molecule.
        substruct_smiles (str): The SMILES of the substructure molecule.

    Returns:
        bool: True if the substructure molecule is a substructure of the PROTAC molecule, False otherwise.
    """
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    substruct_mol = Chem.MolFromSmarts(substruct_smiles)
    return protac_mol.HasSubstructMatch(substruct_mol)


def split_prediction(
        pred: str,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> dict[str, str] | None:
    """ Split a PROTAC SMILES prediction into its three substructures.

    Args:
        pred (str): The SMILES of the PROTAC molecule.
        poi_attachment_id (int): The attachment point ID for the POI substructure.
        e3_attachment_id (int): The attachment point ID for the E3 substructure.

    Returns:
        dict[str, str] | None: A dictionary (with keys: 'e3', 'linker', 'poi') containing the SMILES notations for the POI, linker, and E3 substructures, or None if the prediction is invalid
    """
    ret = {k: None for k in ['poi', 'linker', 'e3']}
    substructs = pred.split('.')
    if len(substructs) != 3:
        return ret
    for substr in substructs:
        if f'[*:{poi_attachment_id}]' in substr and f'[*:{e3_attachment_id}]' not in substr:
            ret['poi'] = substr
        elif f'[*:{e3_attachment_id}]' in substr and f'[*:{poi_attachment_id}]' not in substr:
            ret['e3'] = substr
        elif f'[*:{poi_attachment_id}]' in substr and f'[*:{e3_attachment_id}]' in substr:
            ret['linker'] = substr
    return ret


def check_substructs(
        protac_smiles: str,
        poi_smiles: str = None,
        linker_smiles: str = None,
        e3_smiles: str = None,
        return_bond_types: bool = False,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        pred: str = None,
) -> bool | Tuple[bool, dict[str, str]]:
    """ Check if the reassembled PROTAC is correct.
    
    Args:
        protac_smiles (str): The SMILES of the PROTAC molecule.
        poi_smiles (str): The SMILES of the POI ligand.
        linker_smiles (str): The SMILES of the linker.
        e3_smiles (str): The SMILES of the E3 binder.
        return_bond_types (bool): If True, return the bond types used for the reassembly.
        poi_attachment_id (int): The label of the attachment point for the POI ligand, i.e., "[*:{poi_attachment_id}]".
        e3_attachment_id (int): The label of the attachment point for the E3 binder, i.e., "[*:{e3_attachment_id}]".
        pred (str): The SMILES of the predicted PROTAC molecule.

    Returns:
        bool | Tuple[bool, dict[str, str]]: True if the reassembled PROTAC is correct, False otherwise. If return_bond_types is True, also return the bond types used for the reassembly.
    """
    def get_failed_return():
        if return_bond_types:
            return False, {}
        return False

    # Make some checks before starting and fail if necessary
    all_subs_none = all(v is None for v in [poi_smiles, linker_smiles, e3_smiles])
    any_subs_none = any(v is None for v in [poi_smiles, linker_smiles, e3_smiles])

    if pred is not None and all_subs_none:
        # Split the prediction into the substructures
        pred_substructs = split_prediction(pred, poi_attachment_id, e3_attachment_id)
        if any(v is None for v in pred_substructs.values()):
            return get_failed_return()
        poi_smiles = pred_substructs['poi']
        linker_smiles = pred_substructs['linker']
        e3_smiles = pred_substructs['e3']
    elif pred is None and any_subs_none: 
        return get_failed_return()
    elif pred is None and all_subs_none:
        logging.warning("Arguments 'pred' and 'poi_smiles', 'linker_smiles', 'e3_smiles' cannot be all None.")
        return get_failed_return()

    if f"[*:{poi_attachment_id}]" in e3_smiles:
        return get_failed_return()
    if f"[*:{e3_attachment_id}]" in poi_smiles:
        return get_failed_return()
    if f"[*:{poi_attachment_id}]" not in linker_smiles:
        return get_failed_return()
    if f"[*:{e3_attachment_id}]" not in linker_smiles:
        return get_failed_return()
    
    correct_substructs = False
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    protac_inchi = Chem.MolToInchi(protac_mol)
    protac_smiles_canon = canonize_smiles(protac_smiles)
    bond_types = {}
    bonds = ['single', 'double', 'triple']
    # for e3_bond_type, poi_bond_type in itertools.product([bonds, bonds]):
    for e3_bond_type in bonds:
        for poi_bond_type in bonds:
            try:
                assmbl_smiles, assmbl_mol = reassemble_protac(
                    poi_smiles,
                    linker_smiles,
                    e3_smiles,
                    e3_bond_type,
                    poi_bond_type,
                    poi_attachment_id,
                    e3_attachment_id,
                )
                if assmbl_mol is not None:
                    # If either the InChI or SMILES of the reassembled PROTAC is
                    # the same as the original PROTAC, then the reassembly is
                    # correct.
                    if protac_inchi == Chem.MolToInchi(assmbl_mol):
                        correct_substructs = True
                        bond_types['e3_bond_type'] = e3_bond_type
                        bond_types['poi_bond_type'] = poi_bond_type
                        break
                    if protac_smiles_canon == canonize_smiles(assmbl_smiles):
                        correct_substructs = True
                        bond_types['e3_bond_type'] = e3_bond_type
                        bond_types['poi_bond_type'] = poi_bond_type
                        break
            except:
                continue
    if return_bond_types:
        return correct_substructs, bond_types
    return correct_substructs


def score_prediction(
        protac_smiles: str,
        label_smiles: str,
        pred_smiles: str,
        rouge = None,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=11, fpSize=2048),
        compute_rdkit_metrics: bool = False,
        compute_graph_metrics: bool = False,
        graph_edit_kwargs: Dict[str, Any] = {},
) -> dict[str, float]:
    """ Score a PROTAC SMILES prediction.

    Args:
        protac_smiles (str): The SMILES of the PROTAC molecule.
        label_smiles (str): The SMILES of the ground truth PROTAC molecule.
        pred_smiles (str): The SMILES of the predicted PROTAC molecule.
        rouge (Rouge | None): The Rouge object to use for scoring. If None, do not compute Rouge scores. Example: `rouge = evaluate.load("rouge")`
        poi_attachment_id (int): The attachment point ID for the POI substructure.
        e3_attachment_id (int): The attachment point ID for the E3 substructure.

    Returns:
        dict[str, float]: A dictionary containing the scores for the prediction
    """
    scores = {}

    scores['has_three_substructures'] = has_three_substructures(pred_smiles)
    scores['has_all_attachment_points'] = has_all_attachment_points(pred_smiles)
    scores['tanimoto_similarity'] = 0.0 # Default value
    scores['valid'] = False
    scores['reassembly'] = False
    scores['reassembly_nostereo'] = False

    pred_substructs = split_prediction(pred_smiles, poi_attachment_id, e3_attachment_id)

    # Compute metrics for the "entire" predicted PROTAC molecule
    if all(v is not None for v in pred_substructs.values()):
        scores['valid'] = is_valid_smiles(pred_smiles)
        scores['reassembly'] = check_substructs(
            protac_smiles=protac_smiles,
            poi_smiles=pred_substructs['poi'],
            linker_smiles=pred_substructs['linker'],
            e3_smiles=pred_substructs['e3'],
        )
        if scores['valid']:
            scores['reassembly_nostereo'] = check_substructs(
                protac_smiles=remove_stereo(protac_smiles),
                poi_smiles=remove_stereo(pred_substructs['poi']),
                linker_smiles=remove_stereo(pred_substructs['linker']),
                e3_smiles=remove_stereo(pred_substructs['e3']),
            )
        if scores['valid'] and compute_rdkit_metrics and fpgen is not None:
            # Get Tanimoto similarity between the predicted PROTAC and the ground truth PROTAC
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            label_mol = Chem.MolFromSmiles(label_smiles)
            pred_fp = fpgen.GetFingerprint(pred_mol)
            label_fp = fpgen.GetFingerprint(label_mol)
            scores['tanimoto_similarity'] = DataStructs.TanimotoSimilarity(pred_fp, label_fp)

    if rouge is not None:
        rouge_output = rouge.compute(predictions=[pred_smiles], references=[label_smiles])
        scores.update({k: v for k, v in rouge_output.items()})

    # Compute metrics for each substructure
    label_substructs = split_prediction(label_smiles, poi_attachment_id, e3_attachment_id)

    for sub in ['e3', 'poi', 'linker']:
        # Set default values
        scores[f'{sub}_valid'] = False
        scores[f'{sub}_equal'] = False
        scores[f'{sub}_has_attachment_point(s)'] = False
        scores[f'{sub}_tanimoto_similarity'] = 0.0
        # NOTE: The graph edit distance can be very high and dependant on the
        # graphs, but when the molecule is not valid, then we cannot compute it.
        # Because of that, we instead set it to something very large, in case we
        # need to sum the eval metrics.
        scores[f'{sub}_graph_edit_distance'] = 1e64
        scores[f'{sub}_graph_edit_distance_norm'] = 1.0

        # Skip if the predicted substructure is None from `split_prediction`
        pred_sub = pred_substructs[sub]
        label_sub = label_substructs[sub]
        if pred_sub is None:
            continue

        # Check if the predicted substructure is a valid RDKit molecule
        scores[f'{sub}_valid'], sub_mol = is_valid_smiles(pred_sub, return_mol=True)

        # Check if the predicted substructure has the correct attachment point(s)
        if sub == 'e3':
            if f'[*:{e3_attachment_id}]' in pred_sub and f'[*:{poi_attachment_id}]' not in pred_sub:
                scores[f'{sub}_has_attachment_point(s)'] = True
        elif sub == 'poi':
            if f'[*:{poi_attachment_id}]' in pred_sub and f'[*:{e3_attachment_id}]' not in pred_sub:
                scores[f'{sub}_has_attachment_point(s)'] = True
        elif sub == 'linker':
            if f'[*:{poi_attachment_id}]' in pred_sub and f'[*:{e3_attachment_id}]' in pred_sub:
                scores[f'{sub}_has_attachment_point(s)'] = True
        
        # Check if the predicted substructure InChI is the same as the ground truth substructure InChI
        if scores[f'{sub}_valid']:
            # scores[f'{sub}_equal'] = Chem.MolToInchi(sub_mol) == Chem.MolToInchi(Chem.MolFromSmiles(label_sub))
            canon_pred = canonize_smiles(pred_sub)
            canon_label = canonize_smiles(label_sub)
            scores[f'{sub}_equal'] = canon_pred == canon_label
        
        # Compute graph-related metrics
        if scores[f'{sub}_valid'] and compute_graph_metrics:
            scores[f'{sub}_graph_edit_distance'] = get_smiles2graph_edit_distance(pred_sub, label_sub, **graph_edit_kwargs)
            scores[f'{sub}_graph_edit_distance_norm'] = get_smiles2graph_edit_distance_norm(
                smi1=pred_sub,
                smi2=label_sub,
                ged_G1_G2=scores[f'{sub}_graph_edit_distance'],
                **graph_edit_kwargs,
            )

        # Get Tanimoto similarity b/w the predicted substructure and the ground truth
        if scores[f'{sub}_valid'] and compute_rdkit_metrics:
            pred_mol = Chem.MolFromSmiles(pred_sub)
            label_mol = Chem.MolFromSmiles(label_sub)
            pred_fp = fpgen.GetFingerprint(pred_mol)
            label_fp = fpgen.GetFingerprint(label_mol)
            scores[f'{sub}_tanimoto_similarity'] = DataStructs.TanimotoSimilarity(pred_fp, label_fp)

        # Compute Rouge scores
        if rouge is not None:
            rouge_output = rouge.compute(predictions=[pred_sub], references=[label_sub])
            scores.update({f'{sub}_{k}': v for k, v in rouge_output.items()})

    return scores


def same_atom_counts_and_types(smiles1, smiles2, get_atoms_diff=False):
    """
    Check if two molecules have the same number and types of atoms.

    Args:
    smiles1 (str): SMILES of the first molecule.
    smiles2 (str): SMILES of the second molecule.

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