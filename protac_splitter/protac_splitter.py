from typing import Tuple, Optional, Dict
import logging

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
)
from rdkit import Chem

from .evaluation import (
    split_prediction,
    check_substructs,
)


def dummy2query(mol: Chem.Mol) -> Chem.Mol:
    """ Converts dummy atoms to query atoms, so that a molecule with attachment points can be used in HasSubstructMatch.
    
    Args:
        mol: The molecule to convert.

    Returns:
        The molecule with dummy atoms converted to query atoms
    """
    p = Chem.AdjustQueryParameters.NoAdjustments()
    p.makeDummiesQueries = True
    return Chem.AdjustQueryProperties(mol, p)


def fix_prediction(
        protac_smiles: str,
        pred_smiles: str,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        remove_stereochemistry: bool = False,
) -> Optional[Dict[str, str]]:
    """ Fixes a prediction by replacing the substructure that does not match the PROTAC with the rest of the PROTAC.
    
    Args:
        protac_smiles: The SMILES of the PROTAC.
        pred_smiles: The SMILES of the prediction.
        poi_attachment_id: The attachment point id of the POI. Default is 1.
        e3_attachment_id: The attachment point id of the E3 ligase. Default is 2.

    Returns:
        A dictionary (with keys: 'e3', 'linker', 'poi') containing the fixed substructures, or None if the prediction is invalid.
    """
    
    substructs = split_prediction(pred_smiles)

    # If there are at least two None values, there's nothing we can do to fix it
    if sum(v is None for v in substructs.values()) >= 2:
        logging.warning(f'Invalid prediction for "{pred_smiles}"')
        return None
    
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    substructs = {k: {'smiles': v, 'mol': Chem.MolFromSmiles(v) if v is not None else v} for k, v in substructs.items()}

    # TODO: Check if removing stereochemistry results in a valid prediction
    if remove_stereochemistry:
        Chem.RemoveStereochemistry(protac_mol)
        protac_smiles = Chem.MolToSmiles(protac_mol, canonical=True)
        for k, v in substructs.items():
            if v['mol'] is not None:
                Chem.RemoveStereochemistry(v['mol'])
                substructs[k]['smiles'] = Chem.MolToSmiles(v['mol'], canonical=True)
    
    if all(v['mol'] is not None for v in substructs.values()):
        if check_substructs(
            protac_smiles,
            poi_smiles=substructs['poi']['smiles'],
            linker_smiles=substructs['linker']['smiles'],
            e3_smiles=substructs['e3']['smiles'],
        ):
            return {k: v['smiles'] for k, v in substructs.items()}

    # Check if any of the substructures is NOT a substructure of the PROTAC
    num_matches = 0
    wrong_substruct = None
    for sub in ['poi', 'linker', 'e3']:
        if substructs[sub]['mol'] is None:
            substructs[sub]['match'] = False
            wrong_substruct = sub
        elif protac_mol.HasSubstructMatch(dummy2query(substructs[sub]['mol'])):
            substructs[sub]['match'] = True
            num_matches += 1
        else:
            substructs[sub]['match'] = False
            wrong_substruct = sub

    if num_matches < 2:
        logging.warning(f'Prediction does not contain at least two matching substructures of the PROTAC. Num matches: {num_matches}. Prediction SMILES: "{pred_smiles}"')
        return None

    if num_matches == 3:
        logging.warning(f'Prediction already contains all matching substructures of the PROTAC. Prediction SMILES: "{pred_smiles}"')
        return {k: v['smiles'] for k, v in substructs.items()}

    # Get the order, i.e., either E3 or POI first, based on their size
    if substructs['poi']['mol'] is None or substructs['e3']['mol'] is None:
        logging.warning(f'Invalid prediction for "{pred_smiles}"')
        return None
    
    if substructs['poi']['mol'].GetNumAtoms() > substructs['e3']['mol'].GetNumAtoms():
        order = ['poi', 'e3', 'linker']
    else:
        order = ['e3', 'poi', 'linker']

    fixed_mol = protac_mol
    for sub in order:
        if substructs[sub]['match']:
            fixed_mol = Chem.ReplaceCore(
                fixed_mol,
                dummy2query(substructs[sub]['mol']),
                labelByIndex=False,
                replaceDummies=False,
            )
            if fixed_mol is None:
                logging.warning(f'Failed to replace substructure "{sub}" in prediction SMILES: "{pred_smiles}"')
                return None
            
            # TODO: Try again with another order if when replacing the core we
            # obtain TWO molecules instead of one. This might happen when a
            # substructure is still matching but it is "smaller" than the right
            # one, resulting in "dangling" atoms.

            # Rename the attachment points
            attachment_id = poi_attachment_id if sub == 'poi' else e3_attachment_id
            fixed_smiles = Chem.MolToSmiles(fixed_mol, canonical=True)
            fixed_smiles = fixed_smiles.replace('[1*]', f'[*:{attachment_id}]')
            fixed_smiles = fixed_smiles.replace('[2*]', f'[*:{attachment_id}]')
            fixed_mol = Chem.MolFromSmiles(fixed_smiles)

    if len(fixed_smiles.split('.')) > 1:
        # Get the longest sub-string in fixed_smiles
        fixed_smiles = max(fixed_smiles.split('.'), key=len)

    substructs[wrong_substruct]['smiles'] = fixed_smiles

    if not check_substructs(
        protac_smiles,
        poi_smiles=substructs['poi']['smiles'],
        linker_smiles=substructs['linker']['smiles'],
        e3_smiles=substructs['e3']['smiles'],
    ):
        return None

    return {k: v['smiles'] for k, v in substructs.items()}


def split_protac(
        protac_smiles: str,
        llm_pipeline: Optional[str] = None,
        use_llm_prediction: bool = True,
) -> Tuple[str, str, str]:
    """
    Split a PROTAC SMILES into the two ligands and the linker.

    Dummy implementation that simply splits the SMILES string by '.'.

    Args:
        protac_smiles (str): A string containing the SMILES of the PROTAC.
    
    Returns:
        Tuple[str, str, str]: A tuple containing the SMILES of the first ligand, the linker, and the second ligand.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_llm_prediction:
        if llm_pipeline is None:
            tokenizer = AutoTokenizer.from_pretrained("ailab-bio/PROTAC-Splitter-standard_rand_recombined-ChemBERTa-zinc-base")
            llm_pipeline = llm_pipeline = pipeline(
                "text2text-generation",
                model="ailab-bio/PROTAC-Splitter-standard_rand_recombined-ChemBERTa-zinc-base",
                tokenizer=tokenizer,
                device=device,
            )
        return llm_pipeline(protac_smiles)['generated_text']


    # Split the PROTAC SMILES into the two ligands and the linker
    ligand1, linker, ligand2 = protac_smiles.split('.')
    return ligand1, linker, ligand2