import os
from collections import defaultdict
import logging
import sys
import argparse
from typing import Tuple, Optional, Dict

import pandas as pd

from protac_splitter.evaluation import (
    # is_valid_smiles,
    # has_three_substructures,
    # has_all_attachment_points,
    split_prediction,
    check_substructs,
    score_prediction,
)

import evaluate
from rdkit import Chem
from rdkit import RDLogger
from rdkit import rdBase
from rdkit.Chem import rdFMCS
import networkx as nx


import torch
import numpy as np
from jsonargparse import CLI
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, pipeline, GenerationConfig
from transformers.pipelines.pt_utils import KeyDataset
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


def get_smiles_nostereo(smiles: str) -> str:
    """ Removes stereochemistry from a SMILES string.
    
    Args:
        smiles: The SMILES string to remove stereochemistry from.

    Returns:
        The SMILES string without stereochemistry.
    """
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


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
        A dictionary containing the fixed substructures, or None if the prediction is invalid.
    """
    
    substructs = split_prediction(pred_smiles)

    # If there are at least two None values, there's nothing we can do to fix it
    if sum(v is None for v in substructs.values()) >= 2:
        logging.warning(f'Invalid prediction for "{pred_smiles}"')
        return None
    
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    substructs = {k: {'smiles': v, 'mol': Chem.MolFromSmiles(v)} for k, v in substructs.items()}

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
        logging.warning(f'Prediction does not contain at least two substructures of the PROTAC. Num matches: {num_matches}. Prediction SMILES: "{pred_smiles}"')
        return None

    if num_matches == 3:
        logging.warning(f'Prediction already contains all matching substructures of the PROTAC. Prediction SMILES: "{pred_smiles}"')
        return {k: v['smiles'] for k, v in substructs.items()}

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
            
            # TODO: Try again with another order if when replacing the core we
            # obtain TWO molecules instead of one. This might happen when a
            # substructure is still matching but it is "smaller" than the right
            # one, resulting in "dangling" atoms.

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
        hub_token: Optional[str] = None,
        model_name: str = "ailab-bio/PROTAC-Splitter-standard_recombined-ChemBERTa-zinc-base-v1",
        batch_size: int = 64,
        log_dir: str = 'logs',
        num_proc: int = 8,
        geration_config_kwargs: Optional[Dict[str, any]] = None,
        force_recompute: bool = False,
):
    # Set log level to ERROR
    logging.basicConfig(level=logging.ERROR)

    # Check if hub_token is provided
    if hub_token is None:
        hub_token = os.getenv('HF_TOKEN', None)
        if hub_token is None:
            raise ValueError('Hugging Face API token not provided. Please provide a token using the --hub_token argument or set the HF_TOKEN environment variable')
    
    print('Loading dataset...')
    ds = load_dataset('ailab-bio/PROTAC-Splitter-Dataset', 'standard', token=hub_token)
    test_ds = ds['test']

    # Create a different test dataset with removed stereochemistry
    print('Removing stereochemistry from test dataset...')
    test_ds_nostereo = test_ds.map(
        lambda x: {
            'text': get_smiles_nostereo(x['text']),
            'labels': get_smiles_nostereo(x['labels']),
        },
        num_proc=num_proc,
    )

    # Remove duplicates from test_ds_nostereo
    test_df_nostereo = test_ds_nostereo.to_pandas()
    test_df_nostereo = test_df_nostereo.drop_duplicates(subset=['text'])
    test_ds_nostereo = Dataset.from_pandas(test_df_nostereo, preserve_index=False)

    # Create logs directory if not exists and setup filenames
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name.split('/')[-1].replace('PROTAC-Splitter', 'logs')
    pred_name = model_name.split('/')[-1].replace('PROTAC-Splitter', 'preds')
    metrics_name = model_name.split('/')[-1].replace('PROTAC-Splitter', 'metrics')
    log_filename = os.path.join(log_dir, f'{log_name}.log')
    pred_filename = os.path.join(log_dir, f'{pred_name}.csv')
    metrics_filename = os.path.join(log_dir, f'{metrics_name}.csv')

    # Load predictions if already generated
    if os.path.exists(pred_filename) and not force_recompute:
        print('Loading predictions from file...')
        with open(pred_filename, 'r') as f:
            input_preds = f.readlines()
        preds = [line.split(',')[1].strip() for line in input_preds[1:]]
    else:
        print(f'Pre-generated predictions file "{pred_filename}" not found. Generating predictions...')

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hub_token)
        if geration_config_kwargs is None:
            pipe = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=tokenizer,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                token=hub_token,
            )
        else:
            generation_config = GenerationConfig(
                max_length=512,
                max_new_tokens=512,
                **geration_config_kwargs,
            )
            pipe = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=tokenizer,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                token=hub_token,
                generation_config=generation_config,
            )

        # Generate predictions
        preds = []
        for pred in tqdm(pipe(KeyDataset(test_ds, 'text'), batch_size=batch_size), total=len(ds) // batch_size):
            preds.append(pred[0]['generated_text'])

        inputs = []
        for text in KeyDataset(test_ds, 'text'):
            inputs.append(text)

        # Save text predictions to file
        with open(pred_filename, 'w') as f:
            f.write('input,prediction\n')
            for text, pred in zip(inputs, preds):
                f.write(f'{text},{pred}\n')
        
        # Generate predictions
        preds_nostereo = []
        for pred in tqdm(pipe(KeyDataset(test_ds_nostereo, 'text'), batch_size=batch_size), total=len(ds) // batch_size):
            preds_nostereo.append(pred[0]['generated_text'])

        inputs_nostereo = []
        for text in KeyDataset(test_ds_nostereo, 'text'):
            inputs_nostereo.append(text)

        # Save text predictions to file
        with open(pred_filename.replace('preds', 'preds-nostereo'), 'w') as f:
            f.write('input,prediction\n')
            for text, pred in zip(inputs_nostereo, preds_nostereo):
                f.write(f'{text},{pred}\n')

    print('Predictions collected. Evaluating predictions...')

    # Add `preds` to the test dataset
    test_ds = test_ds.add_column('preds', preds)
    test_ds_nostereo = test_ds_nostereo.add_column('preds', preds_nostereo)

    rouge = evaluate.load("rouge")

    def get_scores(sample):
        protac_smiles = sample['text']
        label_smiles = sample['labels']
        pred_smiles = sample['preds']

        scores = score_prediction(
            protac_smiles,
            label_smiles,
            pred_smiles,
            rouge=rouge,
            poi_attachment_id=1, # Default ones...
            e3_attachment_id=2, # Default ones...
            compute_graph_metrics=True,
            graph_edit_kwargs={'timeout': 1},
        )

        if scores['reassembly']:
            scores['fix_reassembly'] = True
            scores['fix_reassembly_nostereo'] = True
        else:
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
            scores['fix_reassembly'] = fix_reassembly

            # Try to fix prediction by removing stereochemistry
            substructs = fix_prediction(protac_smiles, pred_smiles, remove_stereochemistry=True)
            if substructs is None:
                fix_reassembly = False
            else:
                fix_reassembly = check_substructs(
                    protac_smiles,
                    substructs['poi'],
                    substructs['linker'],
                    substructs['e3'],
                )
            scores['fix_reassembly_nostereo'] = fix_reassembly

        scores['protac_smiles'] = protac_smiles
        scores['label_smiles'] = label_smiles
        scores['pred_smiles'] = pred_smiles

        return scores
    
    # Evaluate predictions
    metrics = test_ds.map(get_scores, num_proc=num_proc, remove_columns=['text', 'labels', 'preds'])
    metrics = metrics.to_pandas()
    metrics_nostereo = test_ds_nostereo.map(get_scores, num_proc=num_proc, remove_columns=['text', 'labels', 'preds'])
    metrics_nostereo = metrics_nostereo.to_pandas()

    # Save metrics to CSV
    metrics.to_csv(metrics_filename, index=False)
    metrics_nostereo.to_csv(metrics_filename.replace('metrics', 'metrics-nostereo'), index=False)

    # Select non-smiles metrics
    non_smiles_metrics = metrics.drop(columns=['protac_smiles', 'label_smiles', 'pred_smiles'])
    non_smiles_metrics_nostereo = metrics_nostereo.drop(columns=['protac_smiles', 'label_smiles', 'pred_smiles'])

    # Print out average metrics
    print('-' * 80)
    for k, v in non_smiles_metrics.mean().items():
        print(f'{k}: {v}')
    print('-' * 80)
    print('No stereochemistry:')
    print('-' * 80)
    for k, v in non_smiles_metrics_nostereo.mean().items():
        print(f'{k}: {v}')
    print('-' * 80)


if __name__ == '__main__':
    # Setup arg parser
    parser = argparse.ArgumentParser(description='Evaluate PROTAC-Splitter models.')
    parser.add_argument('--hub_token', type=str, required=True, help='Hugging Face API token')
    parser.add_argument('--model_name', type=str, default="ailab-bio/PROTAC-Splitter-standard_recombined-ChemBERTa-zinc-base-v1", help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs and predictions')
    parser.add_argument('--num_proc', type=int, default=8, help='Number of processes to use for evaluation')
    parser.add_argument('--force_recompute', action='store_true', help='Force recompute predictions')
    args = parser.parse_args()
    main(
        hub_token=args.hub_token,
        model_name=args.model_name,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        num_proc=args.num_proc,
        force_recompute=args.force_recompute,
    )