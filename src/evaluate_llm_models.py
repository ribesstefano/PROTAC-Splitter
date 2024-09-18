from collections import defaultdict
import logging
import sys
from typing import Tuple, Optional, Dict


from protac_splitter.evaluation import (
    is_valid_smiles,
    has_three_substructures,
    has_all_attachment_points,
    check_substructs,
)
from protac_splitter.llms.evaluation import split_prediction

import evaluate
from rdkit import Chem
from rdkit import RDLogger
from rdkit import rdBase
from rdkit.Chem import rdFMCS
import networkx as nx


import torch
import numpy as np
from jsonargparse import CLI
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm


# Disable the RDKit warnings that pop up when RDKit fails to create molecules
RDLogger.DisableLog("rdApp.*")
blocker = rdBase.BlockLogs()


def smiles2graph(smiles: str) -> nx.Graph:
    # NOTE: https://github.com/maxhodak/keras-molecules/pull/32/files
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        # Skip non-heavy atoms
        if atom.GetAtomicNum() != 0:
            G.add_node(atom.GetIdx(), label=atom.GetSymbol())
    for bond in mol.GetBonds():
        # Skip bonds to non-heavy atoms
        if bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0:
            continue
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=bond.GetBondType())
    return G


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
) -> Optional[Dict[str, str]]:
    """ Fixes a prediction by replacing the substructure that does not match the PROTAC with the rest of the PROTAC.
    
    Args:
        protac_smiles: The SMILES of the PROTAC.
        pred_smiles: The SMILES of the prediction.
        poi_attachment_id: The attachment point id of the POI. Default is 1.
        e3_attachment_id: The attachment point id of the E3 ligase. Default is 2.

    Returns:
        A dictionary containing the fixed substructures, or None if the prediction is invalid.
    """
    
    substructs = split_prediction(pred_smiles)

    if substructs is None:
        logging.warning(f'Invalid prediction for "{pred_smiles}"')
        return None
    
    if check_substructs(
        protac_smiles,
        substructs['poi'],
        substructs['linker'],
        substructs['e3'],
    ):
        return substructs
    
    # TODO: Check if removing stereochemistry results in a valid prediction
    
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    substructs = {k: {'smiles': v, 'mol': Chem.MolFromSmiles(v)} for k, v in substructs.items()}

    # Check if any of the substructures is NOT a substruction of the PROTAC
    num_matches = 0
    wrong_substruct = None
    for sub in ['poi', 'linker', 'e3']:
        if substructs[sub]['mol'] is None:
            return None
        elif protac_mol.HasSubstructMatch(dummy2query(substructs[sub]['mol'])):
            substructs[sub]['match'] = True
            num_matches += 1
        else:
            substructs[sub]['match'] = False
            wrong_substruct = sub

    if num_matches < 2:
        logging.warning(f'Prediction does not contain at least two substructures of the PROTAC. Num matches: {num_matches}. Prediction SMILES: "{pred_smiles}"')
        return None

    if num_matches == 3:
        logging.warning(f'Prediction contains all substructures of the PROTAC. Prediction SMILES: "{pred_smiles}"')
        return None

    fixed_mol = protac_mol
    for sub in ['poi', 'e3', 'linker']:
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

            # Rename the attachment points
            attachment_id = poi_attachment_id if sub == 'poi' else e3_attachment_id
            fixed_smiles = Chem.MolToSmiles(fixed_mol, canonical=True)
            fixed_smiles = fixed_smiles.replace('[1*]', f'[*:{attachment_id}]')
            fixed_mol = Chem.MolFromSmiles(fixed_smiles)

    if len(fixed_smiles.split('.')) > 1:
        # Get the longest sub-string in fixed_smiles
        fixed_smiles = max(fixed_smiles.split('.'), key=len)

    substructs[wrong_substruct]['smiles'] = fixed_smiles

    return {k: v['smiles'] for k, v in substructs.items()}


def main(
        hub_token: str,
        batch_size: int = 64,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "ailab-bio/PROTAC-Splitter-standard_rand_recombined-ChemBERTa-zinc-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hub_token)
    pipe = pipeline(
        "text2text-generation",
        model=model_name, # "ailab-bio/PROTAC-Splitter-standard-ChemBERTa-zinc-base",
        tokenizer=tokenizer,
        device=device,
        token=hub_token,
    )
    ds = load_dataset('ailab-bio/PROTAC-Splitter-Dataset', 'standard', token=hub_token)

    # Load predictions if already generated
    try:
        with open('predictions.txt', 'r') as f:
            preds = f.readlines()
    except FileNotFoundError as e:
        preds = []
        for pred in tqdm(pipe(KeyDataset(ds['test'], 'text'), batch_size=batch_size), total=len(ds) // batch_size):
            preds.append(pred[0]['generated_text'])

        # Save text predictions to file
        with open('predictions.txt', 'w') as f:
            for pred in preds:
                f.write(pred + '\n')

    print('Predictions collected. Evaluating predictions...')

    metrics = defaultdict(list)
    for (protac_smiles, label_smiles, pred_smiles) in tqdm(zip(ds['test']['text'], ds['test']['labels'], preds), total=len(preds)):
        split_pred = split_prediction(pred_smiles)
        if split_pred is None:
            metrics['reassembly'].append(False)
        else:
            metrics['reassembly'].append(check_substructs(protac_smiles, split_pred['poi'], split_pred['linker'], split_pred['e3']))

        # FindMCS: maximum common substructure (MCS) search method that allows atom and/or bond mismatches in the substructures shared among two small molecules.
        split_label = split_prediction(label_smiles)
        for sub in ['poi', 'linker', 'e3']:
            label_mol = Chem.MolFromSmiles(split_label[sub])
            label_graph = smiles2graph(split_label[sub])
            max_edit_dist = label_graph.number_of_edges() + label_graph.number_of_nodes()

            if split_pred is None or split_label is None:
                metrics[f'findmcs_{sub}'].append(0)
                metrics[f'graph_edit_{sub}'].append(max_edit_dist)
                continue
            sub_mol = Chem.MolFromSmiles(split_pred[sub])
            if sub_mol is None:
                metrics[f'findmcs_{sub}'].append(0)
                metrics[f'graph_edit_{sub}'].append(max_edit_dist)
                continue
            mcs = rdFMCS.FindMCS(
                [sub_mol, label_mol],
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                matchValences=True,
            )
            metrics[f'findmcs_{sub}'].append(mcs.numAtoms)
            metrics[f'graph_edit_{sub}'].append(
                nx.graph_edit_distance(smiles2graph(split_pred[sub]), label_graph, timeout=2)
            )

        substructs = fix_prediction(protac_smiles, pred_smiles)
        if substructs is None:
            fix_reassembly = False
        else:
            fix_reassembly = check_substructs(
                protac_smiles,
                substructs['poi'],
                substructs['linker'],
                substructs['e3'],
            )
        metrics['fix_reassembly'].append(fix_reassembly)

    rouge = evaluate.load("rouge")
    for k, v in rouge.compute(predictions=preds, references=ds['test']['labels']).items():
        metrics[k].append(v)

    for k, v in metrics.items():
        print(f'{k}: {np.mean(v)}')


if __name__ == '__main__':
    CLI(main)