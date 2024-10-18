import os
import argparse
import logging

import pandas as pd
import torch
import numpy as np
from jsonargparse import CLI
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, pipeline, GenerationConfig
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from rdkit import RDLogger, rdBase, Chem

from protac_splitter import fix_prediction
from protac_splitter.evaluation import check_substructs, split_prediction

# Disable the RDKit warnings that pop up when RDKit fails to create molecules
RDLogger.DisableLog("rdApp.*")
blocker = rdBase.BlockLogs()


def main(
        model_name: str,
        infile: str,
        outfile: str,
        predfile: str = None,
        batch_size: int = 64,
        infile_protac_col_name: str = 'PROTAC SMILES',
        num_proc: int = 4,
        verbose: int = 0,
):
    if verbose > 0:
        logging.basicConfig(level=logging.DEBUG)
    else:
        # Disable warnings
        logging.basicConfig(level=logging.ERROR)

    hub_token = os.getenv("HF_TOKEN")

    print(f'Loading dataset...')
    # dataset = load_dataset('csv', data_files=infile, split='train')
    df = pd.read_csv(infile)
    dataset = Dataset.from_pandas(df)

    # Load the predictions from the file, one per line
    if predfile is not None and os.path.exists(predfile):
        with open(predfile, 'r') as f:
            preds = f.readlines()
            preds = [p.strip() for p in preds]
    else:
        print(f'Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hub_token)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Loading pipeline on {device.upper()}...')
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=tokenizer,
            device=device,
            token=hub_token,
        )
        print(f'Predicting...')
        preds = []
        for pred in tqdm(pipe(KeyDataset(dataset, infile_protac_col_name), batch_size=batch_size), total=len(dataset) // batch_size):
            preds.append(pred[0]['generated_text'])

        if predfile is not None:
            with open(predfile, 'w') as f:
                for pred in preds:
                    f.write(pred + '\n')
    
    dataset = dataset.add_column('prediction', preds)

    def check_and_fix_pred(sample):
        failed_sample = {
            'PROTAC SMILES': sample[infile_protac_col_name],
            'prediction': sample['prediction'],
            'fixed': False,
            'correct': False,
        }
        pred_mol = Chem.MolFromSmiles(sample['prediction'], sanitize=True)
        if pred_mol is None:
            return failed_sample
        try:
            reassembled_mol = Chem.molzip(pred_mol)
            if reassembled_mol is None:
                return failed_sample
        except:
            return failed_sample
        reassembled_smiles = Chem.MolToSmiles(reassembled_mol, canonical=True)
        if reassembled_smiles == sample[infile_protac_col_name]:
            return {
                'PROTAC SMILES': sample[infile_protac_col_name],
                'prediction': sample['prediction'],
                'fixed': None,
                'correct': True,
            }
        return failed_sample


        # matching = check_substructs(
        #     protac_smiles=sample[infile_protac_col_name],
        #     pred=sample['prediction'],
        # )
        # if not matching:
        #     # Fix the prediction
        #     fixed_substructs = fix_prediction(
        #         protac_smiles=sample[infile_protac_col_name],
        #         pred_smiles=sample['prediction'],
        #     )
        #     if fixed_substructs is not None:
        #         return {
        #             'PROTAC SMILES': sample[infile_protac_col_name],
        #             'prediction': f"{fixed_substructs['e3']}.{fixed_substructs['linker']}.{fixed_substructs['poi']}",
        #             'fixed': True,
        #             'correct': True,
        #         }
        #     return {
        #         'PROTAC SMILES': sample[infile_protac_col_name],
        #         'prediction': sample['prediction'],
        #         'fixed': False,
        #         'correct': False,
        #     }
        # return {
        #     'PROTAC SMILES': sample[infile_protac_col_name],
        #     'prediction': sample['prediction'],
        #     'fixed': None,
        #     'correct': True,
        # }
    
    print(f'Checking and fixing predictions...')
    dataset = dataset.map(check_and_fix_pred, num_proc=num_proc)

    # Save the dataset to CSV
    dataset.to_csv(outfile)

    # Convert to Pandas and print some statistics
    dataset = dataset.to_pandas()
    print(f"Correct predictions: {dataset['correct'].sum()}/{len(dataset)} ({dataset['correct'].sum() / len(dataset) * 100:.2f}%)")
    print(f"Fixed predictions: {dataset['fixed'].sum()}/{len(dataset)} ({dataset['fixed'].sum() / len(dataset) * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PROTAC-Splitter models.')
    parser.add_argument('--model_name', type=str, default="ailab-bio/PROTAC-Splitter-standard_recombined-ChemBERTa-zinc-base-v1", help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--outfile', type=str, default='predictions.json', help='Output file')
    parser.add_argument('--infile', type=str, default='data.csv', help='Input file')
    parser.add_argument('--predfile', type=str, default=None, help='File to load predictions from')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of processes for parallel map processing')
    parser.add_argument('--infile_protac_col_name', type=str, default='PROTAC SMILES', help='Name of the column containing the PROTAC SMILES in the input file')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        batch_size=args.batch_size,
        outfile=args.outfile,
        infile=args.infile,
        predfile=args.predfile,
        num_proc=args.num_proc,
        infile_protac_col_name=args.infile_protac_col_name,
        verbose=args.verbose,
    )