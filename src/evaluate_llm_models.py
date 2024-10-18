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
from protac_splitter import fix_prediction

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


# REF: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/generation/configuration_utils.py#L71
GENERATION_STRATEGY_PARAMS = {
    "greedy": {"num_beams": 1, "do_sample": False},
    "contrastive_search": {"penalty_alpha": 0.1, "top_k": 10},
    "multinomial_sampling": {"num_beams": 1, "do_sample": True},
    "beam_search_decoding": {"num_beams": 5, "do_sample": False},
    "beam_search_multinomial_sampling": {"num_beams": 5, "do_sample": True},
    "diverse_beam_search_decoding": {"num_beams": 5, "num_beam_groups": 5, "diversity_penalty": 1.0},
}

def get_generation_config(generation_strategy: str) -> GenerationConfig:

    return GenerationConfig(
        max_length=512,
        max_new_tokens=512,
        **GENERATION_STRATEGY_PARAMS[generation_strategy],
    )


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

    # Create logs directory if not exists and setup filenames
    os.makedirs(log_dir, exist_ok=True)
    log_name = model_name.split('/')[-1].replace('PROTAC-Splitter', 'logs')
    pred_name = model_name.split('/')[-1].replace('PROTAC-Splitter', 'preds')
    metrics_name = model_name.split('/')[-1].replace('PROTAC-Splitter', 'metrics')
    log_filename = os.path.join(log_dir, f'{log_name}.log')
    pred_filename = os.path.join(log_dir, f'{pred_name}.csv')
    metrics_filename = os.path.join(log_dir, f'{metrics_name}.csv')

    preds = defaultdict(list)

    print(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hub_token)

    print('Loading pipeline for "default" predictions...')
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        token=hub_token,
    )
    print('Generating "default" predictions (training config)...')
    for pred in tqdm(pipe(KeyDataset(test_ds, 'text'), batch_size=batch_size), total=len(test_ds) // batch_size):
        preds['default'].append(pred[0]['generated_text'])

    for generation_strategy in GENERATION_STRATEGY_PARAMS.keys():
        print(f'Loading pipeline for {generation_strategy}...')
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=tokenizer,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            token=hub_token,
            generation_config=get_generation_config(generation_strategy),
        )
        print(f'Generating predictions with generation strategy: {generation_strategy}')
        for pred in tqdm(pipe(KeyDataset(test_ds, 'text'), batch_size=batch_size), total=len(test_ds) // batch_size):
            preds[generation_strategy].append(pred[0]['generated_text'])

    # Define a function to get scores to be mapped to the predictions dataset
    def get_scores(sample):
        protac_smiles = sample['text']
        label_smiles = sample['labels']
        pred_smiles = sample['preds']

        scores = score_prediction(
            protac_smiles,
            label_smiles,
            pred_smiles,
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

        scores['protac_smiles'] = sample['text']
        scores['label_smiles'] = sample['labels']
        scores['pred_smiles'] = sample['preds']

        return scores

    metrics = {}

    for generation_strategy, predictions in preds.items():
        # Add `preds` to the test dataset, so that we can map the scores to the
        # predictions in parallel
        ds = test_ds.add_column('preds', predictions)
        print(f'Evaluating predictions for {generation_strategy}...')
        scores_ds = ds.map(get_scores, num_proc=num_proc, remove_columns=['text', 'labels', 'preds'])
        metrics[generation_strategy] = scores_ds.to_pandas()

    # Join all metrics on ['text', 'labels'] columns
    all_metrics = metrics['default']
    # Add "_default" suffix to columns that do not contain "smiles"
    all_metrics.columns = [f'{c}_default' if 'smiles' not in c else c for c in all_metrics.columns]

    for generation_strategy, df in metrics.items():
        if generation_strategy == 'default':
            continue
        all_metrics = all_metrics.merge(df, on=['protac_smiles', 'label_smiles'], suffixes=('', f'_{generation_strategy}'))

    # Save metrics to CSV
    all_metrics.to_csv(metrics_filename, index=False)

    # Print out average metrics for each generation strategy
    for generation_strategy, metric in metrics.items():
        metric = metric.drop(columns=[c for c in metric.columns if 'smiles' in c])
        print('-' * 80)
        print(f'Generation strategy: {generation_strategy}')
        print('-' * 80)
        print(metric.mean().round(5).T.to_markdown())


    # # Load predictions if already generated
    # if os.path.exists(pred_filename) and os.path.exists(pred_filename.replace('preds', 'preds-nostereo')) and not force_recompute:
    #     print('Loading predictions from file...')
    #     with open(pred_filename, 'r') as f:
    #         input_preds = f.readlines()
    #     preds = [line.split(',')[1].strip() for line in input_preds[1:]]

    #     print('Loading predictions from file...')
    #     with open(pred_filename.replace('preds', 'preds-nostereo'), 'r') as f:
    #         input_preds = f.readlines()
    #     preds_nostereo = [line.split(',')[1].strip() for line in input_preds[1:]]
    # else:
    #     # Load model and tokenizer
    #     print(f'Loading tokenizer and pipeline...')
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, token=hub_token)
    #     if geration_config_kwargs is None:
    #         pipe = pipeline(
    #             "text2text-generation",
    #             model=model_name,
    #             tokenizer=tokenizer,
    #             device='cuda' if torch.cuda.is_available() else 'cpu',
    #             token=hub_token,
    #         )
    #     else:
    #         generation_config = GenerationConfig(
    #             max_length=512,
    #             max_new_tokens=512,
    #             **geration_config_kwargs,
    #         )
    #         pipe = pipeline(
    #             "text2text-generation",
    #             model=model_name,
    #             tokenizer=tokenizer,
    #             device='cuda' if torch.cuda.is_available() else 'cpu',
    #             token=hub_token,
    #             generation_config=generation_config,
    #         )

    #     # Generate predictions
    #     print('Generating predictions...')
    #     preds = []
    #     for pred in tqdm(pipe(KeyDataset(test_ds, 'text'), batch_size=batch_size), total=len(test_ds) // batch_size):
    #         preds.append(pred[0]['generated_text'])

    #     inputs = []
    #     for text in KeyDataset(test_ds, 'text'):
    #         inputs.append(text)

    #     # Save text predictions to file
    #     print('Saving predictions to file...')
    #     with open(pred_filename, 'w') as f:
    #         f.write('input,prediction\n')
    #         for text, pred in zip(inputs, preds):
    #             f.write(f'{text},{pred}\n')
        
    #     # Generate predictions with removed stereochemistry
    #     print('Generating predictions with removed stereochemistry...')
    #     preds_nostereo = []
    #     for pred in tqdm(pipe(KeyDataset(test_ds_nostereo, 'text'), batch_size=batch_size), total=len(test_ds) // batch_size):
    #         preds_nostereo.append(pred[0]['generated_text'])

    #     inputs_nostereo = []
    #     for text in KeyDataset(test_ds_nostereo, 'text'):
    #         inputs_nostereo.append(text)

    #     # Save text predictions to file
    #     print('Saving predictions with removed stereochemistry to file...')
    #     with open(pred_filename.replace('preds', 'preds-nostereo'), 'w') as f:
    #         f.write('input,prediction\n')
    #         for text, pred in zip(inputs_nostereo, preds_nostereo):
    #             f.write(f'{text},{pred}\n')

    # print('Predictions collected. Evaluating predictions...')

    # # Add `preds` to the test dataset
    # test_ds = test_ds.add_column('preds', preds)
    # test_ds_nostereo = test_ds_nostereo.add_column('preds', preds_nostereo)

    # rouge = evaluate.load("rouge")

    # def get_scores(sample):
    #     protac_smiles = sample['text']
    #     label_smiles = sample['labels']
    #     pred_smiles = sample['preds']

    #     scores = score_prediction(
    #         protac_smiles,
    #         label_smiles,
    #         pred_smiles,
    #         rouge=rouge,
    #         poi_attachment_id=1, # Default ones...
    #         e3_attachment_id=2, # Default ones...
    #         compute_graph_metrics=True,
    #         graph_edit_kwargs={'timeout': 1},
    #     )

    #     if scores['reassembly']:
    #         scores['fix_reassembly'] = True
    #         scores['fix_reassembly_nostereo'] = True
    #     else:
    #         substructs = fix_prediction(protac_smiles, pred_smiles)
    #         if substructs is None:
    #             fix_reassembly = False
    #         else:
    #             fix_reassembly = check_substructs(
    #                 protac_smiles,
    #                 substructs['poi'],
    #                 substructs['linker'],
    #                 substructs['e3'],
    #             )
    #         scores['fix_reassembly'] = fix_reassembly

    #         # Try to fix prediction by removing stereochemistry
    #         substructs = fix_prediction(protac_smiles, pred_smiles, remove_stereochemistry=True)
    #         if substructs is None:
    #             fix_reassembly = False
    #         else:
    #             fix_reassembly = check_substructs(
    #                 protac_smiles,
    #                 substructs['poi'],
    #                 substructs['linker'],
    #                 substructs['e3'],
    #             )
    #         scores['fix_reassembly_nostereo'] = fix_reassembly

    #     scores['protac_smiles'] = sample['text']
    #     scores['label_smiles'] = sample['labels']
    #     scores['pred_smiles'] = sample['preds']

    #     return scores
    
    # # Evaluate predictions
    # print('Evaluating predictions...')
    # metrics = test_ds.map(get_scores, num_proc=num_proc, remove_columns=['text', 'labels', 'preds'])
    # metrics = metrics.to_pandas()

    # print('Evaluating predictions with removed stereochemistry...')
    # metrics_nostereo = test_ds_nostereo.map(get_scores, num_proc=num_proc, remove_columns=['text', 'labels', 'preds'])
    # metrics_nostereo = metrics_nostereo.to_pandas()

    # # Save metrics to CSV
    # metrics.to_csv(metrics_filename, index=False)
    # metrics_nostereo.to_csv(metrics_filename.replace('metrics', 'metrics-nostereo'), index=False)

    # # Select non-smiles metrics
    # non_smiles_metrics = metrics.drop(columns=['protac_smiles', 'label_smiles', 'pred_smiles'])
    # non_smiles_metrics_nostereo = metrics_nostereo.drop(columns=['protac_smiles', 'label_smiles', 'pred_smiles'])

    # # Print out average metrics
    # print('-' * 80)
    # for k, v in non_smiles_metrics.mean().items():
    #     print(f'{k}: {v}')
    # print('-' * 80)
    # print('No stereochemistry:')
    # print('-' * 80)
    # for k, v in non_smiles_metrics_nostereo.mean().items():
    #     print(f'{k}: {v}')
    # print('-' * 80)


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