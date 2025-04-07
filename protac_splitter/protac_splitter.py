import logging
from typing import Union, Optional, Dict, Any, List

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from tqdm import tqdm

from .evaluation import (
    split_prediction,
    check_reassembly,
    score_prediction,
)
from .chemoinformatics import (
    canonize,
    dummy2query,
    remove_attach_atom,
    remove_dummy_atoms,
)
from .data.curation.substructure_extraction import get_attachment_bonds


def fix_tetrahedral_centers_ligand(
        protac_mol: Chem.Mol,
        ligand_smiles: str,
        attachment_id: int = 1,
) -> str:
    ligand_mol = Chem.MolFromSmiles(ligand_smiles)
    if ligand_mol is None:
        logging.error(f"Invalid ligand SMILES: {ligand_smiles}")
        return None

    ligand_mol = remove_dummy_atoms(ligand_mol)
    ligand_match = protac_mol.GetSubstructMatch(ligand_mol, useChirality=False) # useChirality=True

    # Get bonds to break to separate the POI ligand
    bonds_to_break_poi = get_attachment_bonds(protac_mol, ligand_match)

    # Return if no bonds are found
    if len(bonds_to_break_poi) != 1:
        logging.error('ERROR: Multiple POI attachment bonds')
        return None

    # Break the bonds to isolate the POI ligand
    frag_ligand_mol = Chem.FragmentOnBonds(protac_mol, bonds_to_break_poi, addDummies=True, dummyLabels=[(attachment_id, attachment_id)])

    # Get the fragments resulting from bond breaking
    try:
        frags = Chem.GetMolFrags(frag_ligand_mol, asMols=True, sanitizeFrags=True)
    except Exception as e:
        logging.error(e)
        return None

    # Identify the ligand fragment
    ligand_fragment = None
    for frag in frags:
        if frag.HasSubstructMatch(ligand_mol):
            ligand_fragment = frag
            break
    if ligand_fragment is None:
        logging.error('ERROR: POI fragment not found')

    ligand_fixed = Chem.MolToSmiles(ligand_fragment)
    ligand_fixed = canonize(ligand_fixed.replace(f'[{attachment_id}*]', f'[*:{attachment_id}]'))
    return ligand_fixed


def fix_prediction(
        protac_smiles: str,
        pred_smiles: str,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        remove_stereochemistry: bool = False,
        verbose: int = 0,
) -> Optional[str]:
    """ Fixes a prediction by replacing the substructure that does not match the PROTAC with the rest of the PROTAC.
    
    Args:
        protac_smiles (str): The SMILES of the PROTAC.
        pred_smiles (str): The SMILES of the prediction.
        poi_attachment_id (int): The attachment point id of the POI. Default is 1.
        e3_attachment_id (int): The attachment point id of the E3 ligase. Default is 2.
        verbose (int): The verbosity level. Default is 0.

    Returns:
        A string containing the fixed predictions, or None if the fixing process failed.
    """
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    if protac_mol is None:
        logging.warning(f"Invalid PROTAC SMILES: {protac_smiles}")
        return None
    
    substructs = split_prediction(pred_smiles)

    # If there are at least two None values, there's nothing we can do to fix it
    if sum(v is None for v in substructs.values()) >= 2:
        logging.warning(f'Unable to continue, more than two substructures are not valid for given input: "{pred_smiles}"')
        return None

    # Get molecules of PROTAC and substructures
    substructs = {k: {'smiles': v, 'mol': Chem.MolFromSmiles(v) if v is not None else v} for k, v in substructs.items()}

    # Check if renaming the attachment points might already fix the prediction
    for sub in ['poi', 'e3', 'both']:
        if sub == 'e3':
            if substructs['e3']['smiles'] is None:
                continue
            e3_attempt = substructs['e3']['smiles'].replace(f'[*:{poi_attachment_id}]', f'[*:{e3_attachment_id}]')
            poi_attempt = substructs['poi']['smiles']
        if sub == 'poi':
            if substructs['poi']['smiles'] is None:
                continue
            e3_attempt = substructs['e3']['smiles']
            poi_attempt = substructs['poi']['smiles'].replace(f'[*:{e3_attachment_id}]', f'[*:{poi_attachment_id}]')
        else:
            if substructs['e3']['smiles'] is None or substructs['poi']['smiles'] is None:
                continue
            e3_attempt = substructs['e3']['smiles'].replace(f'[*:{e3_attachment_id}]', f'[*:{poi_attachment_id}]')
            poi_attempt = substructs['poi']['smiles'].replace(f'[*:{poi_attachment_id}]', f'[*:{e3_attachment_id}]')

        protac_attempt = f"{e3_attempt}.{substructs['linker']['smiles']}.{poi_attempt}"
        if check_reassembly(
            protac_smiles,
            protac_attempt,
        ):
            logging.info(f'Input works when renaming attachment points in {sub.title()} substruct. SMILES: "{protac_attempt}"')
            return protac_attempt
    
        # Check if swapping the POI and E3 attachments in the linker might already fix the prediction
        if substructs['linker']['smiles'] is None:
            continue
        linker_attempt = substructs['linker']['smiles']
        linker_attempt = linker_attempt.replace(f'[*:{poi_attachment_id}]', f'[*:DUMMY]')
        linker_attempt = linker_attempt.replace(f'[*:{e3_attachment_id}]', f'[*:{poi_attachment_id}]')
        linker_attempt = linker_attempt.replace(f'[*:DUMMY]', f'[*:{e3_attachment_id}]')

        # Try with the original POI and E3 substructures
        protac_attempt = f"{substructs['e3']['smiles']}.{linker_attempt}.{substructs['poi']['smiles']}"
        if check_reassembly(
            protac_smiles,
            protac_attempt,
        ):
            logging.info(f'Input works when swapping POI and E3 attachment points in the linker. Fixed SMILES: "{protac_attempt}"')
            return protac_attempt

        # Try with the swapped POI and E3 substructures
        protac_attempt = f"{e3_attempt}.{linker_attempt}.{poi_attempt}"
        if check_reassembly(
            protac_smiles,
            protac_attempt,
        ):
            logging.info(f'Input works when swapping POI and E3 attachment points in the linker and in {sub.title()} substruct. Fixed SMILES: "{protac_attempt}"')
            return protac_attempt

    # Check if removing stereochemistry results in a valid prediction
    if remove_stereochemistry:
        Chem.RemoveStereochemistry(protac_mol)
        protac_smiles = Chem.MolToSmiles(protac_mol, canonical=True)
        for k, v in substructs.items():
            if v['mol'] is not None:
                Chem.RemoveStereochemistry(v['mol'])
                substructs[k]['smiles'] = Chem.MolToSmiles(v['mol'], canonical=True)

    if all(v['mol'] is not None for v in substructs.values()):
        if check_reassembly(
            protac_smiles,
            '.'.join([v['smiles'] for v in substructs.values()]),
        ):
            logging.info(f'Input works when removing stereochemistry. SMILES: "{pred_smiles}"')
            return f"{substructs['e3']['smiles']}.{substructs['linker']['smiles']}.{substructs['poi']['smiles']}"

    # Check if any of the substructures is NOT a substructure of the PROTAC, if
    # so, we mark it as the wrong substructure to fix.
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

    # If the wrong substructure is still matching in the PROTAC, we need to a
    # more complex approach to fix the prediction (see below).
    def remove_substructure(mol, substructure, attachment_id, replaceDummies=False):
        if mol is None or substructure is None:
            return None
        smaller_mol = Chem.ReplaceCore(
            mol,
            substructure,
            labelByIndex=False,
            replaceDummies=replaceDummies,
        )
        if smaller_mol is None:
            logging.warning(f'Failed to remove substructure from prediction SMILES: "{pred_smiles}"')
            return None
        smaller_smiles = Chem.MolToSmiles(smaller_mol, canonical=True)
        smaller_smiles = smaller_smiles.replace('[1*]', f'[*:{attachment_id}]')
        smaller_smiles = smaller_smiles.replace('[2*]', f'[*:{attachment_id}]')
        smaller_mol = canonize(Chem.MolFromSmiles(smaller_smiles))
        return smaller_mol

    # If we still have 3 matches: for each substructure, we progressively remove
    # the other substructures, then we check if the resulting molecule is valid
    # and has only one fragment.
    if num_matches == 3:
        wrong_substruct = None
        for sub in ['poi', 'linker', 'e3']:
            removed_mol = Chem.MolFromSmiles(protac_smiles)

            # Put the current substructure at the end of the list [poi, e3, linker]
            sub_names = ['poi', 'e3', 'linker']
            sub_names.remove(sub)
            sub_names.append(sub)
            # The linker often matches in many parts of the PROTAC, so we remove
            # it when checking the E3 and POI substructures.
            if sub != 'linker':
                sub_names.remove('linker')

            for s in sub_names:
                attachment_id = poi_attachment_id if s == 'poi' else e3_attachment_id
                removed_mol = remove_substructure(
                    removed_mol,
                    dummy2query(substructs[s]['mol']),
                    attachment_id=attachment_id,
                )

            # Check if resulting molecule is None, if so, it is the wrong one
            if removed_mol is None:
                substructs[sub]['match'] = False
                wrong_substruct = sub
                num_matches -= 1
                break

            # Count the number of fragments in the removed molecule
            num_fragments = Chem.GetMolFrags(removed_mol, asMols=True, sanitizeFrags=False)
            if len(num_fragments) > 1:
                substructs[sub]['match'] = False
                wrong_substruct = sub
                num_matches -= 1
                break

    if num_matches == 3:
        logging.warning(f'Prediction already contains all matching substructures of the PROTAC. Prediction SMILES: "{pred_smiles}"')
        return None

    # Get the order in which to remove the substructures and get the final one
    # as the fixed molecule.
    if wrong_substruct == 'linker':
        poi_atoms = substructs['poi']['mol'].GetNumAtoms()
        e3_atoms = substructs['e3']['mol'].GetNumAtoms()
        order = ['poi', 'e3'] if poi_atoms > e3_atoms else ['e3', 'poi']
    else:
        if wrong_substruct == 'poi':
            order = ['e3', 'linker']
        else:
            order = ['poi', 'linker']

    logging.debug(f'Wrong substructure: {wrong_substruct.upper()}. Order: {order}')

    fixed_mol = protac_mol
    for sub in order:
        logging.debug(f'Removing substructure {sub.upper()} from PROTAC.')

        if 'linker' not in order:
            fixed_attach_id = poi_attachment_id if sub == 'poi' else e3_attachment_id
        else:
            fixed_attach_id = poi_attachment_id if 'e3' in order else e3_attachment_id

        if sub == 'linker':
            attach_id = poi_attachment_id if wrong_substruct == 'poi' else e3_attachment_id
            fixed_attach_id = poi_attachment_id if wrong_substruct == 'poi' else e3_attachment_id
            query_mol = remove_attach_atom(substructs[sub]['mol'], attach_id)
            replaceDummies = True
        else:
            query_mol = dummy2query(substructs[sub]['mol'])
            replaceDummies = False

        if verbose:
            # display(Draw.MolToImage(fixed_mol, legend=f"Starting molecule", size=(800, 300)))
            # display(Draw.MolToImage(query_mol, legend=f"Molecule {sub.upper()} to remove", size=(800, 300)))
            pass

        fixed_mol_tmp = remove_substructure(
            fixed_mol,
            query_mol,
            attachment_id=fixed_attach_id,
            replaceDummies=replaceDummies,
        )
        if fixed_mol_tmp is None:
            logging.debug(f'Failed to replace substructure "{sub}" in prediction SMILES: "{pred_smiles}"')
            continue

        fixed_mol = fixed_mol_tmp

        # If there are multiple fragments, keep the biggest one
        fragments = Chem.GetMolFrags(fixed_mol, asMols=True)
        if len(fragments) > 1:
            logging.debug(f'Fixed molecule contains more than one fragment. Keeping the biggest one.')
            max_frag = max(fragments, key=lambda x: x.GetNumAtoms())
            fixed_mol = max_frag

    # Get the SMILES of the fixed molecule
    fixed_smiles = Chem.MolToSmiles(canonize(fixed_mol), canonical=True)
    substructs[wrong_substruct]['smiles'] = fixed_smiles

    if verbose:
        # display(Draw.MolToImage(fixed_mol, legend=f"{wrong_substruct.upper()} fixed molecule: {fixed_smiles}", size=(800, 300)))
        pass

    # Concatenate the substructures check if the re-assembly is correct
    fixed_pred_smiles = f"{substructs['e3']['smiles']}.{substructs['linker']['smiles']}.{substructs['poi']['smiles']}"

    if not check_reassembly(
        protac_smiles,
        fixed_pred_smiles,
    ):
        # Check if by flipping the tetrahedral centers of the ligands we can
        # still fix the prediction.
        protac_mol = canonize(Chem.MolFromSmiles(protac_smiles))
        chiral_centers = Chem.FindMolChiralCenters(
            protac_mol,
            includeUnassigned=True,
            useLegacyImplementation=False,
        )
        if not chiral_centers:
            logging.warning(f"Failed to fix prediction, re-assembly check failed. Generated fixed prediction (failing): {fixed_pred_smiles}")
            return None

        # Get the fixed ligands
        e3_fixed = fix_tetrahedral_centers_ligand(protac_mol, substructs['e3']['smiles'], attachment_id=e3_attachment_id)
        poi_fixed = fix_tetrahedral_centers_ligand(protac_mol, substructs['poi']['smiles'], attachment_id=poi_attachment_id)
        if e3_fixed is None or poi_fixed is None:
            logging.warning(f"Failed to fix prediction, re-assembly check failed. Generated fixed prediction (failing): {fixed_pred_smiles}")
            return None

        # Update the substructures with the fixed ligands and check re-assembly
        substructs['e3']['smiles'] = e3_fixed
        substructs['poi']['smiles'] = poi_fixed
        fixed_pred_smiles = f"{substructs['e3']['smiles']}.{substructs['linker']['smiles']}.{substructs['poi']['smiles']}"
        if not check_reassembly(
            protac_smiles,
            fixed_pred_smiles,
        ):
            logging.warning(f"Failed to fix prediction, re-assembly check failed. Generated fixed prediction (failing): {fixed_pred_smiles}")
            return None

    return fixed_pred_smiles

def split_protac(
        protac_smiles: Union[str, List, pd.DataFrame],
        model_name: str = "ailab-bio/PROTAC-Splitter-Trial-11",
        hf_token: str = None,
        tokenizer: Optional[Any] = None,
        pipe: Optional[Any] = None,
        batch_size: int = 1,
        fix_predictions: bool = True,
        protac_smiles_col: str = "text",
        return_check_reassembly: bool = False,
        verbose: int = 0,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """
    Split a PROTAC SMILES into the two ligands and the linker.

    Args:
        protac_smiles (str | List | pd.DataFrame): The PROTAC SMILES to split.
        model_name (str): The name of the model to use. Default is "ailab-bio/PROTAC-Splitter-Trial-11".
        hf_token (str): The Hugging Face token. Default is None.
        tokenizer (Optional[Any]): The tokenizer to use. Default is None.
        pipe (Optional[Any]): The pipeline to use. Default is None.
        batch_size (int): The batch size to use. Default is 1. From HF: "By default, pipelines will not batch inference for reasons explained in detail here: https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching"
        fix_predictions (bool): Whether to fix the predictions. Default is True.
        protac_smiles_col (str): The column name of the PROTAC SMILES in the DataFrame. Default is "text".
        return_check_reassembly (bool): Whether to return the boolean for each predicted sentence indicating whether re-assemblying the prediction matches the input PROTAC. Default is False.
        verbose (int): The verbosity level. Default is 0.
    
    Returns:
        A dictionary containing the predictions in case of a single PROTAC SMILES, or a list of dictionaries containing the predictions in case of multiple PROTAC SMILES.
    """
    beam_size = 5

    if tokenizer is None:
        if verbose:
            print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    if pipe is None:
        if verbose:
            print("Loading pipeline for \"default\" predictions...")
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            token=hf_token,
            num_return_sequences=beam_size,
        )

    if isinstance(protac_smiles, str):
        protac_smiles_canon = canonize(protac_smiles)
        if protac_smiles_canon is None:
            raise ValueError(f"Invalid PROTAC SMILES: {protac_smiles}")
        pred = pipe(protac_smiles_canon)
        pred = {f"default_pred_n{i}": pred[i]["generated_text"] for i in range(len(pred))}
        if fix_predictions:
            p_fixed = {k: fix_prediction(protac_smiles_canon, v, verbose=verbose) for k, v in pred.items()}
            # For each prediction, if the fixed prediction is not None, we
            # replace the original prediction with the fixed one.
            for k, v in p_fixed.items():
                if v is not None:
                    pred[k] = v
        preds = [pred]

    if isinstance(protac_smiles, list):
        # Canonize and check if all PROTAC SMILES are valid
        protac_smiles_canon = [canonize(protac) for protac in protac_smiles]
        if None in protac_smiles_canon:
            wrong_protacs = [protac for protac, canon in zip(protac_smiles, protac_smiles_canon) if canon is None]
            raise ValueError(f"Invalid PROTAC SMILES in list: {wrong_protacs}")

        # Get the predictions for all PROTAC SMILES
        preds = pipe(protac_smiles_canon, batch_size=batch_size)
        preds = [{f"default_pred_n{i}": p["generated_text"] for i, p in enumerate(pred)} for pred in preds]

        if fix_predictions:
            for i, (protac, pred) in enumerate(zip(protac_smiles_canon, preds)):
                p_fixed = {k: fix_prediction(protac, v, verbose=verbose) for k, v in pred.items()}
                # For each prediction, if the fixed prediction is not None, we
                # replace the original prediction with the fixed one.
                for k, v in p_fixed.items():
                    if v is not None:
                        preds[i][k] = v

    if isinstance(protac_smiles, pd.DataFrame):
        # Check if the DataFrame contains a columns named `protac_smiles_col`
        if protac_smiles_col not in protac_smiles.columns:
            raise ValueError(f"DataFrame must contain a column named \"{protac_smiles_col}\".")
        
        # Canonize and check if all PROTAC SMILES are valid
        protac_smiles_canon = protac_smiles.apply(lambda x: canonize(x[protac_smiles_col]), axis=1)

        # Check if there are invalid PROTAC SMILES
        if protac_smiles_canon.isnull().any():
            wrong_protacs = protac_smiles[protac_smiles_canon.isnull()]
            raise ValueError(f"Invalid PROTAC SMILES in DataFrame: {wrong_protacs}")

        # Convert the Series to a DataFrame
        protac_smiles_canon = pd.DataFrame(protac_smiles_canon, columns=[protac_smiles_col])
        
        # Convert the DataFrame to a Dataset
        dataset = Dataset.from_pandas(protac_smiles_canon)
        preds = []
        for pred in tqdm(pipe(KeyDataset(dataset, protac_smiles_col), batch_size=batch_size), total=len(dataset) // batch_size, desc="Generating predictions"):
            p = {f"default_pred_n{i}": pred[i]["generated_text"] for i in range(len(pred))}
            preds.append(p)

        if fix_predictions:
            for i, (protac, pred) in tqdm(enumerate(zip(protac_smiles_canon, preds)), desc="Fixing predictions", total=len(preds)):
                p_fixed = {k: fix_prediction(protac, v, verbose=verbose) for k, v in pred.items()}
                # For each prediction, if the fixed prediction is not None, we
                # replace the original prediction with the fixed one.
                for k, v in p_fixed.items():
                    if v is not None:
                        pred[k] = v

    if return_check_reassembly:
        if isinstance(protac_smiles_canon, str):
            protac_smiles_list = [protac_smiles_canon]
        elif isinstance(protac_smiles_canon, list):
            protac_smiles_list = protac_smiles_canon
        elif isinstance(protac_smiles_canon, pd.DataFrame):
            protac_smiles_list = protac_smiles_canon[protac_smiles_col].tolist()
        
        print("Checking re-assembly...")
        for protac, pred in zip(protac_smiles_list, preds):
            for i in range(beam_size):
                pred[f"reassembly_correct_n{i}"] = check_reassembly(protac, pred[f"default_pred_n{i}"])

        # Just take the first prediction if the input was a string
        if isinstance(protac_smiles, str):
            preds = preds[0]

    return preds