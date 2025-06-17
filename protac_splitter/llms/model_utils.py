from typing import Optional, List, Dict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import torch


def get_encoder_decoder_model(
        pretrained_encoder: str = "seyonec/ChemBERTa-zinc-base-v1",
        pretrained_decoder: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_length: Optional[int] = 512,
        tie_encoder_decoder: bool = False,
) -> EncoderDecoderModel:
    """ Get the EncoderDecoderModel model for the PROTAC splitter.

    Args:
        pretrained_encoder (str): The pretrained model to use for the encoder. Default: "seyonec/ChemBERTa-zinc-base-v1"
        pretrained_decoder (str): The pretrained model to use for the decoder. Default: "seyonec/ChemBERTa-zinc-base-v1"
        max_length (int): The maximum length of the input sequence. Default: 512
        tie_encoder_decoder (bool): Whether to tie the encoder and decoder weights. Default: False

    Returns:
        EncoderDecoderModel: The EncoderDecoderModel model for the PROTAC splitter
    """
    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
        pretrained_encoder,
        pretrained_decoder,
        tie_encoder_decoder=tie_encoder_decoder,
    )
    print(f"Number of parameters: {bert2bert.num_parameters():,}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_encoder)
    # Tokenizer-related configs
    bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
    bert2bert.config.eos_token_id = tokenizer.sep_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id
    bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size
    # Generation configs
    # NOTE: See full list of configurations can be found here: https://huggingface.co/docs/transformers/v4.33.3/en/main_classes/text_generation#transformers.GenerationConfig
    bert2bert.encoder.config.max_length = max_length
    bert2bert.decoder.config.max_length = max_length

    def setup_gen(config):
        config.do_sample = True
        config.num_beams = 5
        config.top_k = 20
        config.max_length = 512
        # config.max_new_tokens = 512
        return config
    
    bert2bert.config = setup_gen(bert2bert.config)
    bert2bert.encoder.config = setup_gen(bert2bert.encoder.config)
    bert2bert.decoder.config = setup_gen(bert2bert.decoder.config)
    bert2bert.decoder.config.is_decoder = True
    bert2bert.generation_config = setup_gen(bert2bert.generation_config)
    
    # bert2bert.config.do_sample = True
    # bert2bert.config.num_beams = 5
    # bert2bert.config.top_k = 20
    # bert2bert.config.max_length=512
    # bert2bert.config.max_new_tokens=512

    # bert2bert.generation_config.max_new_tokens = 512
    # bert2bert.generation_config.min_new_tokens = 512
    
    # bert2bert.config.max_new_tokens = 514
    # bert2bert.config.early_stopping = True
    # bert2bert.config.length_penalty = 2.0
    # # bert2bert.config.no_repeat_ngram_size = 3 # Default: 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert2bert.to(device)

    return bert2bert


def get_causal_model(
        pretrained_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_length: Optional[int] = 512,
) -> AutoModelForCausalLM:
    """ Get the causal language model for the PROTAC splitter.

    Args:
        pretrained_model (str): The pretrained model to use for the causal language model. Default: "seyonec/ChemBERTa-zinc-base-v1"
        max_length (int): The maximum length of the input sequence. Default: 512

    Returns:
        AutoModelForCausalLM: The causal language model for the PROTAC splitter
    """
    model = AutoModelForCausalLM.from_pretrained(pretrained_model, is_decoder=True)
    # model.is_decoder = True # It might not be necessary, but it's good to be explicit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


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
        smiles_column: str = 'prompt',
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
    for pred in tqdm(pipe(KeyDataset(test_ds, smiles_column), batch_size=batch_size, max_length=512), total=len(test_ds) // batch_size):
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
        smiles_column: str = 'text',
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
    for pred in tqdm(pipe(KeyDataset(test_ds, smiles_column), batch_size=batch_size, max_length=512), total=len(test_ds) // batch_size):
        p = {f'pred_n{i}': p['generated_text'] for i, p in enumerate(pred)}
        preds.append(p)
    return preds

def run_pipeline(
        pipe: pipeline,
        test_ds: Dataset,
        batch_size: int,
        is_causal_language_model: bool,
        smiles_column: str = 'text',
) -> List[Dict[str, str]]:
    if is_causal_language_model:
        return run_causal_pipeline(pipe, test_ds, batch_size, smiles_column)
    else:
        return run_seq2seq_pipeline(pipe, test_ds, batch_size, smiles_column)