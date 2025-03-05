""" Evaluation functions for the protac_splitter package. They need to be generic to accomodate predictions coming from different models. """

import re
import logging
from typing import Tuple, Any, Dict, Optional, Union

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs

# Disable RDKit logging: when checking SMILES validity, we suppress warnings
RDLogger.DisableLog("rdApp.*")

from .chemoinformatics import (
    canonize,
    canonize_smiles,
    remove_stereo,
    get_substr_match,
)
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
    """ Check if a PROTAC SMILES has three substructures. """
    return smiles.count(".") == 2


def has_all_attachment_points(smiles: str) -> bool:
    """ Check if a PROTAC SMILES has all attachment points, i.e., [*:1] and [*:2], two each. """
    return smiles.count("[*:1]") == 2 and smiles.count("[*:2]") == 2


def split_prediction(
        pred: str,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> Optional[dict[str, str]]:
    """ Split a PROTAC SMILES prediction into its three substructures.

    Args:
        pred (str): The SMILES of the PROTAC molecule.
        poi_attachment_id (int): The attachment point ID for the POI substructure.
        e3_attachment_id (int): The attachment point ID for the E3 substructure.

    Returns:
        dict[str, str] | None: A dictionary (with keys: 'e3', 'linker', 'poi') containing the SMILES notations for the POI, linker, and E3 substructures, or None if the prediction is invalid
    """
    ret = {k: None for k in ['poi', 'linker', 'e3']}
    ligands = pred.split('.')
    if len(ligands) != 3:
        return ret
    for ligand in ligands:
        if f'[*:{poi_attachment_id}]' in ligand and f'[*:{e3_attachment_id}]' not in ligand:
            ret['poi'] = ligand
        elif f'[*:{e3_attachment_id}]' in ligand and f'[*:{poi_attachment_id}]' not in ligand:
            ret['e3'] = ligand
        elif f'[*:{poi_attachment_id}]' in ligand and f'[*:{e3_attachment_id}]' in ligand:
            ret['linker'] = ligand
    return ret


def rename_attachment_id(mol: Union[str, Chem.Mol], old_id: int, new_id: int) -> Union[str, Chem.Mol]:
    """ Rename an attachment point ID in a molecule.

    Args:
        mol: The input molecule.
        old_id: The old attachment point ID.
        new_id: The new attachment point ID.

    Returns:
        The renamed molecule.
    """
    return_str = False
    if isinstance(mol, Chem.Mol):
        mol = Chem.MolToSmiles(mol, canonical=True)
        return_str = True
    # Regex-replace the patterns "[*:old_id]" or "[old_id*]" with "[*:new_id]"
    mol = re.sub(rf'\[\*:{old_id}\]', f'[*:{new_id}]', mol)
    mol = re.sub(rf'\[{old_id}\*\]', f'[*:{new_id}]', mol)
    mol = canonize_smiles(mol)
    if mol is None:
        return None
    mol = Chem.MolFromSmiles(mol)
    if return_str:
        return Chem.MolToSmiles(mol, canonical=True)
    return mol

def at_least_two_ligands_correct(
        protac_smiles: str,
        ligands_smiles: str,
) -> bool:
    """ Check if at least two ligands are correct. """
    # Check if there is at least one "." in the ligands SMILES
    if "." not in ligands_smiles:
        return False
    ligands = ligands_smiles.split(".")
    return True


def check_reassembly(
        protac_smiles: str,
        ligands_smiles: str,
        stats: Optional[Dict[str, int]] = None,
        linker_can_be_null: bool = False,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        verbose: int = 0,
        return_reassembled_smiles: bool = False,
) -> bool:
    """Check if the reassembled PROTAC matches the original PROTAC SMILES.

    Args:
        protac_smiles (str): The original PROTAC SMILES.
        ligands_smiles (str): The SMILES of the joined PROTAC ligands, separated by a "." (dot).
        stats (Optional[Dict[str, int]]): A dictionary to store statistics about the reassembly process.
        linker_can_be_null (bool): If False, the linker cannot be empty, and if so, a None will be returned. If True, a special check is performed to rename the E3 and WH attchament points to assemble them together.
        poi_attachment_id (int): The label of the attachment point for the POI ligand, i.e., "[*:{poi_attachment_id}]". Default is 1.
        e3_attachment_id (int): The label of the attachment point for the E3 binder, i.e., "[*:{e3_attachment_id}]". Default is 2.
        verbose (int): The verbosity

    Returns:
        bool: True if the reassembled PROTAC matches the original PROTAC SMILES, False otherwise. None if it failed.
    """
    ligands_smiles = canonize_smiles(ligands_smiles)
    if ligands_smiles is None:
        if verbose:
            logging.error('Ligand could be canonicalized.')
        return (False, None) if return_reassembled_smiles else False

    null_linker_e3 = f'[*:{e3_attachment_id}][*:{poi_attachment_id}]'
    null_linker_poi = f'[*:{poi_attachment_id}][*:{e3_attachment_id}]'
    linker_is_null = False
    if null_linker_e3 in ligands_smiles or null_linker_poi in ligands_smiles:
        # If the linker is empty, remove the linker atoms
        ligands_smiles = ligands_smiles.replace(null_linker_poi, '')
        ligands_smiles = ligands_smiles.replace(null_linker_e3, '')
        ligands_smiles = ligands_smiles.replace('..', '.')
        ligands_smiles = ligands_smiles.rstrip('.')
        ligands_smiles = ligands_smiles.lstrip('.')
        ligands_smiles = canonize_smiles(ligands_smiles)
        linker_is_null = True

    if linker_can_be_null or linker_is_null:
        if len(ligands_smiles.split('.')) == 2:
            # Replace the attachment points with a third one (they will be joined later)
            ligands_smiles = rename_attachment_id(ligands_smiles, e3_attachment_id, max([poi_attachment_id, e3_attachment_id]) + 1)
            ligands_smiles = rename_attachment_id(ligands_smiles, poi_attachment_id, max([poi_attachment_id, e3_attachment_id]) + 1)

    ligands_mol = Chem.MolFromSmiles(ligands_smiles)
    if ligands_mol is None:
        if verbose:
            logging.error('ligands_mol is None')
        return (False, None) if return_reassembled_smiles else False

    try:
        reassembled_mol = Chem.molzip(ligands_mol)
        if reassembled_mol is None:
            if stats is not None:
                stats['molzip failed'] += 1
            if verbose:
                logging.error(f'molzip failed')
            return (False, None) if return_reassembled_smiles else False
    except:
        if stats is not None:
            stats['molzip failed (exception)'] += 1
        if verbose:
            logging.error(f'molzip failed (exception)')
        return (False, None) if return_reassembled_smiles else False

    try:
        reassembled_smiles = canonize(Chem.MolToSmiles(reassembled_mol))
        if reassembled_smiles is None:
            if stats is not None:
                stats['MolToSmiles of reassembled failed'] += 1
            if verbose:
                logging.error('MolToSmiles of reassembled failed')
            return (False, None) if return_reassembled_smiles else False
    except:
        if stats is not None:
            stats['MolToSmiles of reassembled failed'] += 1
        if verbose:
            logging.error('MolToSmiles of reassembled failed')
        return (False, None) if return_reassembled_smiles else False

    is_equal = canonize(protac_smiles) == reassembled_smiles

    return (is_equal, reassembled_smiles) if return_reassembled_smiles else is_equal


def check_substructs(
        protac_smiles: str,
        poi_smiles: str = None,
        linker_smiles: str = None,
        e3_smiles: str = None,
        return_bond_types: bool = False,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        pred: str = None,
) -> Union[bool, Tuple[bool, dict[str, str]]]:
    """ DEPRECATED.
    
    Check if the reassembled PROTAC is correct.
    
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
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    protac_num_atoms = protac_mol.GetNumHeavyAtoms()

    scores = {
        'has_three_substructures': has_three_substructures(pred_smiles),
        'has_all_attachment_points': has_all_attachment_points(pred_smiles),
        'num_fragments': pred_smiles.count('.') + 1,
        'tanimoto_similarity': 0.0, # Default value
        'valid': False,
        'reassembly': False,
        'reassembly_nostereo': False,
        'heavy_atoms_difference': protac_num_atoms,
        'heavy_atoms_difference_norm': 1.0,
        'all_ligands_equal': False,
    }

    pred_substructs = split_prediction(pred_smiles, poi_attachment_id, e3_attachment_id)

    # Compute metrics for the "entire" predicted PROTAC molecule
    if None not in list(pred_substructs.values()):
        e3_nostereo = remove_stereo(pred_substructs['e3'])
        linker_nostereo = remove_stereo(pred_substructs['linker'])
        poi_nostereo = remove_stereo(pred_substructs['poi'])
        if None not in [e3_nostereo, linker_nostereo, poi_nostereo]:
            pred_nostereo = f"{e3_nostereo}.{linker_nostereo}.{poi_nostereo}"
            scores['reassembly_nostereo'] = check_reassembly(remove_stereo(protac_smiles), pred_nostereo)

        scores['valid'] = is_valid_smiles(pred_smiles)
        is_equal, reassembled_smiles = check_reassembly(protac_smiles, pred_smiles, return_reassembled_smiles=True)
        scores['reassembly'] = is_equal

        # Get the number of heavy atoms difference between the reassembled PROTAC and the ground truth PROTAC
        if reassembled_smiles is not None:
            reassembled_mol = Chem.MolFromSmiles(reassembled_smiles)
            if reassembled_mol is not None:
                scores['heavy_atoms_difference'] -= reassembled_mol.GetNumHeavyAtoms()
                scores['heavy_atoms_difference_norm'] = scores['heavy_atoms_difference'] / protac_num_atoms

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

    # Set default values
    for sub in ['e3', 'poi', 'linker']:
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
        scores[f'{sub}_heavy_atoms_difference'] = 0
        try:
            scores[f'{sub}_heavy_atoms_difference'] = Chem.MolFromSmiles(label_substructs[sub]).GetNumHeavyAtoms()
        except:
            print(f"WARNING: {sub} substructure is None in the label: '{label_smiles}' - PROTAC: '{protac_smiles}'")
        scores[f'{sub}_heavy_atoms_difference_norm'] = 1.0

    # Calculate metrics for each substructure
    for sub in ['e3', 'poi', 'linker']:
        # Skip if the predicted substructure is None from `split_prediction`
        pred_sub = pred_substructs[sub]
        label_sub = label_substructs[sub]
        if pred_sub is None:
            continue
        if label_sub is None:
            print(f"WARNING: {sub} substructure is None in the label: '{label_smiles}' - PROTAC: '{protac_smiles}'")
            continue

        # Check if the predicted substructure is a valid RDKit molecule
        sub_valid, sub_mol = is_valid_smiles(pred_sub, return_mol=True)
        scores[f'{sub}_valid'] = sub_valid

        if sub_mol is None:
            continue

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
        
        # Get the number of heavy atoms difference between the predicted substructure and the ground truth substructure
        if scores[f'{sub}_valid']:
            pred_mol = Chem.MolFromSmiles(pred_sub)
            scores[f'{sub}_heavy_atoms_difference'] -= pred_mol.GetNumHeavyAtoms()
            scores[f'{sub}_heavy_atoms_difference_norm'] = scores[f'{sub}_heavy_atoms_difference'] / Chem.MolFromSmiles(label_sub).GetNumHeavyAtoms()

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

    scores['all_ligands_equal'] = all([scores[f'{sub}_equal'] for sub in ['e3', 'poi', 'linker']])

    return scores