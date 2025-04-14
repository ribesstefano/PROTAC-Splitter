import os
from collections import defaultdict
import logging
import argparse
from typing import Optional, List, Dict

import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit import rdBase

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, pipeline, GenerationConfig, EncoderDecoderModel
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from torchmetrics.text import Perplexity


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
    "beam_search_decoding": {"num_beams": 5, "do_sample": False, "num_return_sequences": 5},
    "beam_search_multinomial_sampling": {"num_beams": 5, "do_sample": True, "num_return_sequences": 5},
    "diverse_beam_search_decoding": {"num_beams": 5, "num_beam_groups": 5, "diversity_penalty": 1.0, "num_return_sequences": 5},
}

def get_generation_config(generation_strategy: str) -> GenerationConfig:
    """ Get the generation config for the given generation strategy. """
    return GenerationConfig(
        max_length=512,
        max_new_tokens=512,
        **GENERATION_STRATEGY_PARAMS[generation_strategy],
    )

def get_pipeline(
        model_name: str,
        token: str,
        is_causal_language_model: bool,
        generation_strategy: Optional[str] = None,
) -> pipeline:
    if is_causal_language_model and generation_strategy is None:
        print('Loading pipeline for causal language models...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, padding_side='left')
        return pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            token=token,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_return_sequences=1,
        )
    if is_causal_language_model and generation_strategy is not None:
        print('Loading pipeline for causal language models...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, padding_side='left')
        return pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            token=token,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            generation_config=get_generation_config(generation_strategy),
        )
    if not is_causal_language_model and generation_strategy is None:
        print('Loading pipeline for sequence-to-sequence models...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        return pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=tokenizer,
            token=token,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
    if not is_causal_language_model and generation_strategy is not None:
        print('Loading pipeline for sequence-to-sequence models...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        return pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=tokenizer,
            token=token,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            generation_config=get_generation_config(generation_strategy),
        )

def run_causal_pipeline(
        pipe: pipeline,
        test_ds: Dataset,
        batch_size: int,
) -> List[Dict[str, str]]:
    """ Run the pipeline for causal language models and return the predictions.
    
    Args:
        pipe (pipeline): The pipeline object to use for generating predictions.
        test_ds (Dataset): The test dataset to generate predictions for.
        batch_size (int): The batch size to use for generating predictions.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the predictions.
    """
    preds = []
    for pred in tqdm(pipe(KeyDataset(test_ds, 'prompt'), batch_size=batch_size, max_length=512), total=len(test_ds) // batch_size):
        generated_text = [p['generated_text'] for p in pred]
        # Remove the prompt from the generated text
        generated_text = ['.'.join(t.split('.')[1:]) for t in generated_text]
        # Add the predictions to the list
        p = {f'pred_n{i}': t for i, t in enumerate(generated_text)}
        preds.append(p)
    return preds

def run_seq2seq_pipeline(
        pipe: pipeline,
        test_ds: Dataset,
        batch_size: int,
) -> List[Dict[str, str]]:
    """ Run the pipeline for sequence-to-sequence models and return the predictions.
    
    Args:
        pipe (pipeline): The pipeline object to use for generating predictions.
        test_ds (Dataset): The test dataset to generate predictions for.
        batch_size (int): The batch size to use for generating predictions.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the predictions.
    """
    preds = []
    for pred in tqdm(pipe(KeyDataset(test_ds, 'text'), batch_size=batch_size, max_length=512), total=len(test_ds) // batch_size):
        p = {f'pred_n{i}': p['generated_text'] for i, p in enumerate(pred)}
        preds.append(p)
    return preds

def run_pipeline(
        pipe: pipeline,
        test_ds: Dataset,
        batch_size: int,
        is_causal_language_model: bool,
) -> List[Dict[str, str]]:
    if is_causal_language_model:
        return run_causal_pipeline(pipe, test_ds, batch_size)
    else:
        return run_seq2seq_pipeline(pipe, test_ds, batch_size)


def main(
        hub_token: Optional[str] = None,
        model_name: str = "ailab-bio/PROTAC-Splitter-standard_recombined-ChemBERTa-zinc-base-v1",
        batch_size: int = 64,
        log_dir: str = 'logs',
        num_proc: int = 8,
        eval_gen_strategies: bool = False,
        report_model_name: Optional[str] = None,
        cache_dir: str = '~/.cache/huggingface',
        is_causal_language_model: bool = False,
        get_predictions_probabilities: bool = False,
        dataset_dir: Optional[str] = 'ailab-bio/PROTAC-Splitter-Dataset',
        dataset_config: Optional[str] = 'clustered',
        dataset_test_split: Optional[str] = 'held_out',
):
    # Set log level to ERROR
    logging.basicConfig(level=logging.ERROR)

    # Check if hub_token is provided
    if hub_token is None:
        hub_token = os.getenv('HF_TOKEN', None)
        if hub_token is None:
            raise ValueError('Hugging Face API token not provided. Please provide a token using the --hub_token argument or set the HF_TOKEN environment variable')
    
    print('Loading dataset...')
    if os.path.exists(dataset_dir):
        test_ds = load_dataset(
            dataset_dir,
            data_dir=dataset_config,
        )[dataset_test_split]
    else:
        test_ds = load_dataset(
            dataset_dir,
            dataset_config,
            token=hub_token,
            cache_dir=cache_dir,
        )[dataset_test_split]

    if report_model_name is None:
        report_model_name = [n for n in model_name.split('/') if 'PROTAC-Splitter' in n][0]
    print(f'Collecting predictions for model: {model_name}')
    print(f'Model name for reporting: {report_model_name}')

    preds = defaultdict(list)

    print('Loading pipeline for "default" predictions...')
    pipe = get_pipeline(model_name, hub_token, is_causal_language_model)

    # Modify the dataset to add a prompt for causal language models
    if is_causal_language_model:
        print('Modifying dataset for causal language models...')
        test_ds = test_ds.map(
            lambda x: {
                'text': x["text"],
                'prompt': x["text"] + '.',
                'labels': x['labels'],
            },
            num_proc=num_proc,
        )
    
    if get_predictions_probabilities:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hub_token)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # NOTE: We will ignore the padding token for the perplexity calculation
        perplexity = Perplexity(ignore_index=tokenizer.pad_token_id).to(device)

        if is_causal_language_model:
            raise ValueError('Getting predictions probabilities is not supported for causal language models yet.')

        model = EncoderDecoderModel.from_pretrained(model_name, token=hub_token)
        model.to(device)
        model.eval()

        # Apply tokenization and run generate to batches of inputs
        for i in tqdm(range(0, len(test_ds), batch_size), desc='Getting probabilities and perplexity scores'):
            # Get a batch of inputs, tokenize them, and move to GPU
            indeces = list(range(i, min(i + batch_size, len(test_ds))))
            batch = tokenizer(
                test_ds.select(indeces)['text'], # [i:i+batch_size],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Generate predictions and get probabilities
            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Get the probabilities of the generated sequences
            # NOTE: The scores are in log space, so we need to exponentiate them
            # to get the probabilities
            probs = torch.exp(outputs.sequences_scores).tolist()
            
            # Use the generated output as the decoder input IDs
            decoder_input_ids = outputs.sequences
            labels = decoder_input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            # Generate decoder attention mask
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
            decoder_attention_mask[decoder_input_ids == tokenizer.pad_token_id] = 0

            # Compute loss on generated output (we need the logits)
            # NOTE: Since we have an encoder-decoder model, the "inputs" are actually the
            # ones to the encoder. We however need the logits on the decoder outputs, so
            # we need to input them accordingly. The "labels" will be the decoder inputs
            # themselves.
            with torch.no_grad():
                logits = model(
                    **batch,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                ).logits

            # Compute the perplexity score for each generated sequence in batch
            for generated_logits, generated_target, generated_prob in zip(logits, decoder_input_ids, probs):
                generated_string = tokenizer.decode(generated_target, skip_special_tokens=True)
                generated_logits = generated_logits.unsqueeze(0)
                generated_target = generated_target.unsqueeze(0)

                # NOTE: We need to shift both the logits and the target, since
                # each token in the target indexes the logit of the previous
                # token.
                perplexity_score = perplexity(
                    preds=generated_logits[:, :-1],
                    target=generated_target[:, 1:],
                ).item()

                preds['default'].append({
                    'pred_n0': generated_string,
                    'prob_n0': generated_prob,
                    'perplexity_n0': perplexity_score,
                })
    else:
        # Run the pipeline to generate predictions with its predefined gen. strategy
        print('Generating "default" predictions (training config)...')
        preds['default'] = run_pipeline(pipe, test_ds, batch_size, is_causal_language_model)

    if eval_gen_strategies:
        for generation_strategy in GENERATION_STRATEGY_PARAMS.keys():
            print(f'Loading pipeline for {generation_strategy}...')
            pipe = get_pipeline(model_name, hub_token, is_causal_language_model, generation_strategy)

            print(f'Generating predictions with generation strategy: {generation_strategy}')
            preds[generation_strategy] = run_pipeline(pipe, test_ds, batch_size, is_causal_language_model)

    print('Predictions collected. Saving to file...')

    df = []
    for i, (text, labels) in enumerate(zip(test_ds['text'], test_ds['labels'])):
        row = {}
        row['protac_smiles'] = text
        row['label_smiles'] = labels
        for generation_strategy, predictions in preds.items():
            p = {f'{generation_strategy}_{k}': v for k, v in predictions[i].items()}
            row.update(p)
        df.append(row)
    
    # Save predictions to a Pandas DataFrame first, then to a CSV file    
    df = pd.DataFrame(df)
    df['model_name'] = model_name

    # Create logs directory if not exists and setup filenames
    os.makedirs(log_dir, exist_ok=True)

    pred_name = f'{report_model_name}-preds'
    pred_filename = os.path.join(log_dir, f'{pred_name}.csv')
    df.to_csv(pred_filename, index=False)
    print(f'Predictions saved to: {pred_filename}')

    # NOTE: There is no need to evaluate the predictions here and wasting GPU time.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PROTAC-Splitter models.')
    parser.add_argument('--hub_token', type=str, required=True, help='Hugging Face API token')
    parser.add_argument('--model_name', type=str, default="ailab-bio/PROTAC-Splitter-standard_recombined-ChemBERTa-zinc-base-v1", help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs and predictions')
    parser.add_argument('--num_proc', type=int, default=8, help='Number of processes to use for evaluation')
    parser.add_argument('--report_model_name', type=str, help='Model name to use for reporting')
    parser.add_argument('--cache_dir', type=str, default='~/.cache/huggingface', help='Hugging Face cache directory')
    parser.add_argument('--is_causal_language_model', type=lambda x: x.lower() == 'true', default=False, help='Whether the model is a causal language model')
    parser.add_argument('--eval_gen_strategies', type=lambda x: x.lower() == 'true', default=False, help='Whether to evaluate different generation strategies')
    parser.add_argument('--get_predictions_probabilities', type=lambda x: x.lower() == 'true', default=False, help='Whether to get predictions probabilities')
    parser.add_argument('--dataset_dir', type=str, default='ailab-bio/PROTAC-Splitter-Dataset', help='Dataset directory')
    parser.add_argument('--dataset_config', type=str, default='clustered', help='Dataset config')
    parser.add_argument('--dataset_test_split', type=str, default='held_out', help='Dataset test split')
    args = parser.parse_args()
    main(
        hub_token=args.hub_token,
        model_name=args.model_name,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        num_proc=args.num_proc,
        report_model_name=args.report_model_name,
        cache_dir=args.cache_dir,
        is_causal_language_model=args.is_causal_language_model,
        eval_gen_strategies=args.eval_gen_strategies,
        get_predictions_probabilities=args.get_predictions_probabilities,
        dataset_dir=args.dataset_dir,
        dataset_config=args.dataset_config,
        dataset_test_split=args.dataset_test_split,
    )