import os
import evaluate
import math
import torch
from model_utils import get_model
from data_utils import (
    load_tokenized_dataset,
    load_trl_dataset,
    data_collator_for_trl,
)
from evaluation_metrics import compute_metrics_with_chem, reward_function
from tqdm import tqdm
from datasets import load_dataset
from typing import Optional, Literal, Dict
from functools import partial
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EncoderDecoderModel,
    AutoConfig,
)
import huggingface_hub as hf

from .llms.hf_utils import (
    create_hf_repository,
    delete_hf_repository,
)


def train_mlm_model(
    model_name: str,
    ds_name: str = 'ailab-bio/PROTAC-Substructures',
    ds_config: str = 'encoder_mlm_dataset',
    max_steps: int = 2000,
    num_train_epochs: int = -1,
    batch_size: int = 128,
    batch_size_tokenizer: int = 1024,
    gradient_accumulation_steps: int = 4,
    hub_token: Optional[str] = None,
    organization: Optional[str] = None,
    output_dir: str = "./models/",
    mlm_probability: float = 0.15,
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_length: int = 512,
    delete_repo_first: bool = False,
    training_args: Optional[TrainingArguments] = None,
):
    """Trains a masked language model on a given dataset.
    
    Args:
        model_name (str): The name of the model to be trained.
        ds_name (str, optional): The name of the dataset to be used for training. Defaults to 'ailab-bio/PROTAC-Substructures'.
        ds_config (str, optional): The name of the dataset configuration to be used for training. Defaults to 'encoder_mlm_dataset'.
        max_steps (int, optional): The maximum number of training steps. Defaults to 2000.
        num_train_epochs (int, optional): The number of training epochs. Defaults to -1.
        batch_size (int, optional): The batch size. Defaults to 128.
        batch_size_tokenizer (int, optional): The batch size for the tokenizer. Defaults to 1024.
        gradient_accumulation_steps (int, optional): The number of gradient accumulation steps. Defaults to 4.
        hub_token (Optional[str], optional): The Hugging Face token. Defaults to None.
        organization (Optional[str], optional): The Hugging Face organization. Defaults to None.
        output_dir (str, optional): The output directory. Defaults to "./models/".
        mlm_probability (float, optional): The probability of masking tokens. Defaults to 0.15.
        tokenizer (AutoTokenizer | str, optional): The tokenizer. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_model_name (str, optional): The name of the pretrained model. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        max_length (int, optional): The maximum length of the input sequence. Defaults to 512.
        delete_repo_first (bool, optional): Whether to delete the repository first. Defaults to False.
        training_args (Optional[TrainingArguments], optional): The training arguments. Defaults to None.
    """
    if hub_token is not None:
        hf.login(token=hub_token)
    # Setup output directory and Hugging Face repository
    output_dir += f"/{model_name}"
    if organization is not None:
        hub_model_name = f"{organization}/{model_name}"
        if delete_repo_first:
            delete_hf_repository(repo_id=hub_model_name, token=hub_token)
        repo_url = create_hf_repository(
            repo_id=hub_model_name,
            repo_type="model",
            exist_ok=True,
            private=True,
            token=hub_token,
        )
        print(f"Repository '{hub_model_name}' created at URL: {repo_url}")
    else:
        hub_model_name = None
    # Load pretrained MLM model
    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name,
        token=hub_token,
        force_download=True,
        max_length=max_length,
    )
    # Load tokenizer
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            token=hub_token,
        )
    tokenizer.pad_token = tokenizer.eos_token
    # Load and tokenize MLM dataset
    mlm_dataset = load_dataset(
        ds_name,
        ds_config,
        token=hub_token,
    )
    eval_mlm_dataset = load_dataset(
        ds_name,
        "80-20-split",
        split="validation",
        token=hub_token,
    )
    tokenized_mlm_dataset = mlm_dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        batch_size=batch_size_tokenizer,
        remove_columns=["text", "labels"],
    )["train"]
    eval_tokenized_mlm_dataset = eval_mlm_dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        batch_size=batch_size_tokenizer,
        remove_columns=["text", "labels"],
    )
    # Setup trainer
    if training_args is None:
        per_device_batch_size = batch_size // gradient_accumulation_steps
        training_args = TrainingArguments(
            output_dir=output_dir,
            # Optimizer-related configs
            learning_rate=5e-8,
            optim="adamw_torch",
            lr_scheduler_type="linear", # Default: "linear"
            # Batch size and device configs
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            auto_find_batch_size=True,
            # Evaluation and checkpointing configs
            evaluation_strategy="steps",
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            eval_steps=100,
            save_steps=500,
            fp16=True,
            # Logging configs
            log_level="info",
            logging_steps=50,
            disable_tqdm=True,
            # Hub information configs
            push_to_hub=True, # NOTE: Done manually further down
            hub_token=hub_token,
            hub_model_id=hub_model_name,
            hub_strategy="checkpoint", # NOTE: Allows to resume training from last checkpoint 
            hub_private_repo=True,
            # Other configs
            seed=42,
            data_seed=42,
        )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_mlm_dataset,
        eval_dataset=eval_tokenized_mlm_dataset,
        data_collator=data_collator,
    )
    # Get perplexity before training
    eval_results = trainer.evaluate()
    print(f"Perplexity before training: {math.exp(eval_results['eval_loss']):.2f}")
    # Train model
    trainer.train(
        # resume_from_checkpoint=True, # "last-checkpoint",
    )
    # Get perplexity after training
    eval_results = trainer.evaluate()
    print(f"Perplexity after training: {math.exp(eval_results['eval_loss']):.2f}")
    # Push model to Hugging Face Hub
    if hub_model_name is not None:
        trainer.push_to_hub(
            commit_message="Initial version",
            model_name=hub_model_name,
            license="mit",
            finetuned_from=pretrained_model_name,
            tasks=["Fill-Mask"],
            tags=["PROTAC", "cheminformatics"],
            dataset=ds_name,
            dataset_args=ds_config,
        )