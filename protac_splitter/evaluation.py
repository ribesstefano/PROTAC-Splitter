""" Evaluation functions for the protac_splitter package. They need to be generic to accomodate predictions coming from different models. """

import math
import re
import logging
from typing import Tuple, Any, Dict, List, Optional, Union

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, Descriptors, rdMolDescriptors

# Disable RDKit logging: when checking SMILES validity, we suppress warnings
RDLogger.DisableLog("rdApp.*")

from .chemoinformatics import (
    canonize,
    canonize_smiles,
    remove_stereo,
    remove_dummy_atoms,
    get_substr_match,
)
from .protac_cheminformatics import reassemble_protac
from .graphs_utils import (
    get_smiles2graph_edit_distance,
    get_smiles2graph_edit_distance_norm,
)
# NOTE: protac_splitter.graphs.* is imported lazily (inside _known_ligand_similarity)
# rather than at module level: protac_splitter.graphs/__init__.py pulls in
# data.curation.bond_adjustments -> data.curation.curation, which itself imports from
# this module, so an eager import here would be a circular import at package load time.


def is_valid_smiles(
        smiles: Optional[str],
        return_mol: bool = False,
) -> Union[bool, Tuple[bool, Chem.Mol]]:
    """ Check if a SMILES is valid, i.e., it can be parsed by RDKit.
    
    Args:
        smiles (Optional[str]): The SMILES to check.
        return_mol (bool): If True, return the RDKit molecule object, i.e., `(is_valid, mol)`.
    
    Returns:
        bool | Tuple[bool, Chem.Mol]: True if the SMILES is valid, False otherwise. If return_mol is True, also return the RDKit molecule object.
    """
    if smiles is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if return_mol:
        return mol is not None, mol
    return mol is not None


def has_three_substructures(smiles: Optional[str]) -> bool:
    """ Check if a PROTAC SMILES has three substructures. """
    if smiles is None:
        return False
    return smiles.count(".") == 2


def has_all_attachment_points(smiles: Optional[str]) -> bool:
    """ Check if a PROTAC SMILES has all attachment points, i.e., [*:1] and [*:2], two each. """
    if smiles is None:
        return False
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
    if pred is None:
        return ret

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
        'num_fragments': 0 if pred_smiles is None else pred_smiles.count('.') + 1,
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
            logging.warning(f"WARNING: {sub} substructure is None in the label: '{label_smiles}' - PROTAC: '{protac_smiles}'")
        scores[f'{sub}_heavy_atoms_difference_norm'] = 1.0

    # Calculate metrics for each substructure
    for sub in ['e3', 'poi', 'linker']:
        # Skip if the predicted substructure is None from `split_prediction`
        pred_sub = pred_substructs[sub]
        label_sub = label_substructs[sub]
        if pred_sub is None:
            continue
        if label_sub is None:
            logging.warning(f"WARNING: {sub} substructure is None in the label: '{label_smiles}' - PROTAC: '{protac_smiles}'")
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
            label_mol = Chem.MolFromSmiles(label_sub)
            if label_mol is None:
                logging.warning(f"WARNING: {sub} substructure is None in the label: '{label_smiles}' - PROTAC: '{protac_smiles}'")
                continue
            scores[f'{sub}_heavy_atoms_difference'] -= pred_mol.GetNumHeavyAtoms()
            scores[f'{sub}_heavy_atoms_difference_norm'] = scores[f'{sub}_heavy_atoms_difference'] / label_mol.GetNumHeavyAtoms()

        # Get Tanimoto similarity b/w the predicted substructure and the ground truth
        if scores[f'{sub}_valid'] and compute_rdkit_metrics:
            pred_mol = Chem.MolFromSmiles(pred_sub)
            label_mol = Chem.MolFromSmiles(label_sub)
            if label_mol is None:
                logging.warning(f"WARNING: {sub} substructure is None in the label: '{label_smiles}' - PROTAC: '{protac_smiles}'")
                continue
            pred_fp = fpgen.GetFingerprint(pred_mol)
            label_fp = fpgen.GetFingerprint(label_mol)
            scores[f'{sub}_tanimoto_similarity'] = DataStructs.TanimotoSimilarity(pred_fp, label_fp)

        # Compute Rouge scores
        if rouge is not None:
            rouge_output = rouge.compute(predictions=[pred_sub], references=[label_sub])
            scores.update({f'{sub}_{k}': v for k, v in rouge_output.items()})

    scores['all_ligands_equal'] = all([scores[f'{sub}_equal'] for sub in ['e3', 'poi', 'linker']])

    return scores


# ---------------------------------------------------------------------------
# Single-split QC scorer — reference-free plausibility checks that vary with
# *which bonds were cut*, usable both for offline dataset QC
# (protac_splitter.data.curation.dataset_qc) and as an inference-time quality
# gate (protac_splitter.split_protac(model="adaptive")).
# ---------------------------------------------------------------------------

FRAGMENT_MW_BOUNDS: Dict[str, Tuple[float, float]] = {
    "e3": (150.0, 700.0),
    "poi": (120.0, 900.0),
}

# Heavy-atom-count bounds, checked alongside MW: MW alone can be fooled by atom
# composition (e.g. a couple of halogens push MW into range on a fragment with almost
# no skeleton, or the reverse for a fluorine-/PEG-heavy fragment) — a fragment must
# clear *both* bars to be considered plausible. Derived from the heavy-atom-count
# distribution of the curated reference lists themselves (graphs/clustering.py's
# DEFAULT_REPRESENTATIVE_E3S / _WHS): p5/p95 are ~(17, 42) for E3s and ~(15, 65) for
# warheads, widened a bit here to avoid rejecting legitimate small fragments.
FRAGMENT_HEAVY_ATOM_BOUNDS: Dict[str, Tuple[int, int]] = {
    "e3": (10, 55),
    "poi": (8, 70),
}

# Above this many consecutive non-ring, unbranched heavy atoms strung out anywhere off
# an E3/POI fragment's own ring system, the fragment looks less like "a ligand with a
# short tether" and more like "an under-cut linker still fused onto it" (e.g. a
# PEG/amide chain that should have been isolated as part of the linker instead).
# Calibrated against Datasets/smiles/dataset-curated-held-out.csv (5,670 real curated
# PROTAC splits): on the 115 manually-curated rows the max is 4 (POI) / 3 (E3); on the
# full set p99 is 6 (POI) / 3 (E3) and the true max is 15 (POI) / 10 (E3) — 6 sits with
# a 2-unit margin above the trusted max while only trimming ~1% of the noisier,
# non-manually-curated tail.
FRAGMENT_ATTACHMENT_CHAIN_LIMIT = 6


def _ring_system_atoms(mol: Chem.Mol, start_idx: int) -> set:
    """All atoms in the fused/bridged ring system containing `start_idx` (BFS restricted
    to ring bonds), e.g. the full isoindolinone-benzo bicycle, not just one ring of it.
    """
    ring_bonds = set()
    for ring in mol.GetRingInfo().BondRings():
        ring_bonds.update(ring)
    system = {start_idx}
    frontier = [start_idx]
    while frontier:
        idx = frontier.pop()
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            if bond.GetIdx() in ring_bonds:
                other = bond.GetOtherAtomIdx(idx)
                if other not in system:
                    system.add(other)
                    frontier.append(other)
    return system


def _chain_length_from(mol: Chem.Mol, start_idx: int, exclude: set) -> int:
    """Heavy atoms strung out from `start_idx` before hitting a ring or a real branch.
    Terminal decorations (=O, =S, halogens, a lone terminal methyl) don't count as a
    branch — only a neighbor that itself keeps going is a genuine fork in the chain.
    """
    visited = set(exclude)
    current = mol.GetAtomWithIdx(start_idx)
    length = 0
    while current is not None and not current.IsInRing():
        length += 1
        visited.add(current.GetIdx())
        next_atoms = [
            n for n in current.GetNeighbors()
            if n.GetIdx() not in visited and n.GetDegree() > 1
        ]
        if len(next_atoms) != 1:
            break  # dead end (0 real continuations) or a real fork (2+) -- stop counting
        current = next_atoms[0]
    return length


def _attachment_chain_length(mol: Chem.Mol) -> Optional[int]:
    """Longest run of heavy atoms strung off the fragment's ring system before hitting
    another ring or a branch. A long run is the topological signature of a cut placed
    too early: real E3/POI cores are ring-based, so a long acyclic tail anywhere off
    that ring system (not just in line with the attachment point itself — the leak can
    just as easily hang off a different ring substituent) means part of the linker is
    still attached. If the attachment point isn't on a ring at all, this just walks the
    chain from it directly.
    """
    dummy = next((a for a in mol.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None or dummy.GetDegree() == 0:
        return None

    anchor = dummy.GetNeighbors()[0]
    if not anchor.IsInRing():
        return _chain_length_from(mol, anchor.GetIdx(), exclude={dummy.GetIdx()})

    ring_system = _ring_system_atoms(mol, anchor.GetIdx())
    max_length = 0
    for idx in ring_system:
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in ring_system or n_idx == dummy.GetIdx():
                continue
            max_length = max(max_length, _chain_length_from(mol, n_idx, exclude=ring_system))
    return max_length


def _fragment_descriptors(role: str, frag_smiles: Optional[str]) -> Dict[str, Any]:
    # NOTE: descriptors are computed on the dummy-stripped fragment, which RDKit
    # re-perceives valence for on reparse — the attachment carbon gets capped with an
    # implicit H, as if the fragment were synthesized standalone. That's correct for
    # MW/heavy-atom counts.
    out = {
        f"{role}_mw": None,
        f"{role}_heavy_atoms": None,
        f"{role}_disconnected": False,
        f"{role}_attachment_chain_length": None,
        f"flag_{role}_out_of_range": False,
        f"flag_{role}_linker_leak": False,
    }
    if frag_smiles is None:
        return out

    # Chain-length needs the dummy atom itself, so it's computed on the raw fragment
    # before dummy-stripping below.
    raw_mol = Chem.MolFromSmiles(frag_smiles)
    if raw_mol is not None:
        chain_length = _attachment_chain_length(raw_mol)
        out[f"{role}_attachment_chain_length"] = chain_length
        if chain_length is not None:
            out[f"flag_{role}_linker_leak"] = chain_length >= FRAGMENT_ATTACHMENT_CHAIN_LIMIT

    stripped = remove_dummy_atoms(frag_smiles, canonical=True)
    if stripped is None:
        return out
    out[f"{role}_disconnected"] = "." in stripped

    mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        return out

    mw = Descriptors.MolWt(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    out[f"{role}_mw"] = round(mw, 1)
    out[f"{role}_heavy_atoms"] = heavy_atoms

    mw_bounds = FRAGMENT_MW_BOUNDS.get(role)
    atom_bounds = FRAGMENT_HEAVY_ATOM_BOUNDS.get(role)
    mw_out_of_range = mw_bounds is not None and not (mw_bounds[0] <= mw <= mw_bounds[1])
    atom_count_out_of_range = atom_bounds is not None and not (atom_bounds[0] <= heavy_atoms <= atom_bounds[1])
    out[f"flag_{role}_out_of_range"] = mw_out_of_range or atom_count_out_of_range
    return out


def _linker_topology(linker_smiles: Optional[str]) -> Dict[str, Any]:
    out = {
        "linker_heavy_atoms_between": None,
        "linker_branch_points": None,
        "linker_ring_count": None,
        "flag_linker_too_short": False,
        "flag_linker_branchy": False,
    }
    if linker_smiles is None:
        return out

    mol = Chem.MolFromSmiles(linker_smiles)
    if mol is None:
        return out

    dummy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_idx) != 2:
        return out

    path = Chem.GetShortestPath(mol, dummy_idx[0], dummy_idx[1])
    if not path:
        return out

    heavy_between = len(path) - 2  # path includes both dummy endpoints
    # Ring atoms (e.g. a substituted piperazine/piperidine spacer) routinely have
    # degree >= 3 without the linker actually tree-branching — only count branching
    # *outside* rings.
    branch_points = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetIdx() not in dummy_idx and atom.GetDegree() >= 3 and not atom.IsInRing()
    )
    out["linker_heavy_atoms_between"] = heavy_between
    out["linker_branch_points"] = branch_points
    out["linker_ring_count"] = rdMolDescriptors.CalcNumRings(mol)
    out["flag_linker_too_short"] = heavy_between <= 1
    out["flag_linker_branchy"] = branch_points >= 2
    return out


def _known_ligand_similarity(
    e3_smiles: Optional[str],
    poi_smiles: Optional[str],
    e3_sim_threshold: float,
    poi_sim_threshold: float,
    representative_e3s_fp: Optional[List] = None,
    representative_whs_fp: Optional[List] = None,
) -> Dict[str, Any]:
    from .graphs.utils import max_tanimoto_similarity
    from .graphs.clustering import get_representative_e3s_fp, get_representative_whs_fp

    out = {
        "e3_sim_to_known_e3": None,
        "e3_sim_to_known_wh": None,
        "poi_sim_to_known_wh": None,
        "poi_sim_to_known_e3": None,
        "flag_e3_low_similarity": False,
        "flag_poi_low_similarity": False,
        "flag_role_swap_suspected": False,
    }
    representative_e3s_fp = representative_e3s_fp if representative_e3s_fp is not None else get_representative_e3s_fp()
    representative_whs_fp = representative_whs_fp if representative_whs_fp is not None else get_representative_whs_fp()

    e3_stripped = remove_dummy_atoms(e3_smiles, canonical=True) if e3_smiles else None
    poi_stripped = remove_dummy_atoms(poi_smiles, canonical=True) if poi_smiles else None

    if e3_stripped is not None and Chem.MolFromSmiles(e3_stripped) is not None:
        out["e3_sim_to_known_e3"] = round(float(max_tanimoto_similarity(e3_stripped, representative_e3s_fp)), 3)
        out["e3_sim_to_known_wh"] = round(float(max_tanimoto_similarity(e3_stripped, representative_whs_fp)), 3)
        out["flag_e3_low_similarity"] = out["e3_sim_to_known_e3"] < e3_sim_threshold

    if poi_stripped is not None and Chem.MolFromSmiles(poi_stripped) is not None:
        out["poi_sim_to_known_wh"] = round(float(max_tanimoto_similarity(poi_stripped, representative_whs_fp)), 3)
        out["poi_sim_to_known_e3"] = round(float(max_tanimoto_similarity(poi_stripped, representative_e3s_fp)), 3)
        out["flag_poi_low_similarity"] = out["poi_sim_to_known_wh"] < poi_sim_threshold

    if out["e3_sim_to_known_wh"] is not None and out["poi_sim_to_known_e3"] is not None:
        # A swap is only worth flagging if the "wrong-label" resemblance is itself a
        # real match, not just the larger of two noise-level numbers: for a chemotype
        # poorly covered by both reference lists, every similarity can sit under the
        # low-similarity floor, and picking the marginally-larger one is comparing
        # noise to noise, not evidence of a swap.
        similarity_floor = min(e3_sim_threshold, poi_sim_threshold)
        e3_looks_like_wh = (
            out["e3_sim_to_known_wh"] > out["e3_sim_to_known_e3"]
            and out["e3_sim_to_known_wh"] >= similarity_floor
        )
        poi_looks_like_e3 = (
            out["poi_sim_to_known_e3"] > out["poi_sim_to_known_wh"]
            and out["poi_sim_to_known_e3"] >= similarity_floor
        )
        out["flag_role_swap_suspected"] = e3_looks_like_wh and poi_looks_like_e3
    return out


def score_split(
    protac_smiles: str,
    pred: Optional[str],
    e3_sim_threshold: float = 0.2,
    poi_sim_threshold: float = 0.2,
    representative_e3s_fp: Optional[List] = None,
    representative_whs_fp: Optional[List] = None,
    poi_attachment_id: int = 1,
    e3_attachment_id: int = 2,
) -> Dict[str, Any]:
    """Score one candidate split of `protac_smiles` on reference-free plausibility checks.

    Only checks that vary with *which bonds were cut* are included here (structural
    validity/reassembly, fragment size, linker topology, known-ligand similarity) — that
    is what makes the result usable to compare candidates coming from different
    splitting methods/parameters. Checks that depend only on the intact input molecule
    (e.g. BRENK instability, leaving groups) are constant across every candidate split
    of the same PROTAC, so they can't discriminate between candidates; those live in
    `protac_splitter.data.curation.dataset_qc` instead, not here.

    `protac_smiles` is assumed already valid/canonical — callers that can't guarantee
    that should validate it themselves before calling this.

    Returns a flat dict of metrics and `flag_*` booleans, with `flag_structural`
    aggregating the hard gate (valid, 3 substructures, both attachment points,
    reassembles). Pass the result to `count_flags()` for a single (n_flags, reasons)
    summary.
    """
    result: Dict[str, Any] = {}
    pred = None if pred is None or (isinstance(pred, float) and np.isnan(pred)) else pred

    result["pred_valid"] = is_valid_smiles(pred) if pred else False
    result["has_three_substructures"] = has_three_substructures(pred)
    result["has_all_attachment_points"] = has_all_attachment_points(pred)
    result["reassembly_ok"] = bool(check_reassembly(
        protac_smiles, pred, poi_attachment_id=poi_attachment_id, e3_attachment_id=e3_attachment_id,
    )) if pred else False

    frags = split_prediction(pred, poi_attachment_id, e3_attachment_id) if pred else {"e3": None, "linker": None, "poi": None}

    for role in ("e3", "poi"):
        result.update(_fragment_descriptors(role, frags.get(role)))
    result.update(_linker_topology(frags.get("linker")))
    result.update(_known_ligand_similarity(
        frags.get("e3"), frags.get("poi"), e3_sim_threshold, poi_sim_threshold,
        representative_e3s_fp, representative_whs_fp,
    ))

    result["flag_structural"] = (
        not result["pred_valid"]
        or not result["has_three_substructures"]
        or not result["has_all_attachment_points"]
        or not result["reassembly_ok"]
    )
    return result


def count_flags(result: Dict[str, Any]) -> Tuple[int, str]:
    """Summarize a dict's `flag_*` booleans into a count and a ';'-joined reason string."""
    triggered = [k[len("flag_"):] for k, v in result.items() if k.startswith("flag_") and v]
    return len(triggered), ";".join(triggered)