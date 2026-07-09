""" Train a masked language model (MLM) using an encoder-decoder architecture. """
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import subprocess

import torch
import huggingface_hub as hf
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)

from protac_splitter.llms.data_utils import load_tokenized_dataset
from protac_splitter.llms.hf_utils import (
    create_hf_repository,
    delete_hf_repository,
    repo_exists,
)
from protac_splitter.llms.model_utils import get_encoder_decoder_model


def compute_metrics_for_mlm(pred) -> Dict[str, float]:
    """Compute metrics for MLM predictions, i.e., perplexity."""
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids
    
    # Convert to torch tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    
    # Compute masked loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    return {
        "perplexity": torch.exp(loss).item(),
        "loss": loss.item()
    }


def train_mlm_model(
        model_id: str,
        ds_name: str,
        ds_config: str = 'default',
        learning_rate: float = 5e-5,
        max_steps: int = -1,
        num_train_epochs: int = 40,
        batch_size: int = 128,
        batch_size_tokenizer: int = 512,
        gradient_accumulation_steps: int = 4,
        hub_token: Optional[str] = None,
        organization: Optional[str] = None,
        output_dir: str = "./models/",
        tokenizer: Union[AutoTokenizer, str] = "seyonec/ChemBERTa-zinc-base-v1",
        pretrained_encoder: str = "seyonec/ChemBERTa-zinc-base-v1",
        pretrained_decoder: str = "seyonec/ChemBERTa-zinc-base-v1",
        encoder_max_length: int = 512,
        decoder_max_length: int = 512,
        tie_encoder_decoder: bool = False,
        delete_repo_if_exists: bool = False,
        delete_local_repo_if_exists: bool = False,
        training_args: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: Optional[str] = None,
        num_proc_map: int = 1,
        per_device_batch_size: Optional[int] = None,
        lr_scheduler_type: Optional[str] = None,
        mlm_probability: float = 0.15,
        randomize_smiles: bool = False,
        randomize_smiles_prob: float = 0.5,
        randomize_smiles_repeat: int = 1,
):
    """
    Trains a masked language model (MLM) using an encoder-decoder architecture.
  
    Args:
        model_id (str): The name of the model to be trained.
        ds_name (str): The name of the dataset to use for training.
        ds_config (str): The configuration of the dataset to use. Default: 'default'.
        learning_rate (float): The learning rate for training. Default: 5e-5.
        max_steps (int): The maximum number of training steps. Default: -1.
        num_train_epochs (int): The number of training epochs. Default: 40.
        batch_size (int): The total batch size. Default: 128.
        batch_size_tokenizer (int): The batch size for the tokenizer. Default: 512.
        gradient_accumulation_steps (int): The number of gradient accumulation steps. Default: 4.
        hub_token (str): The Hugging Face token for authentication. Default: None.
        organization (str): The organization to push the model to. Default: None.
        output_dir (str): The output directory for the model. Default: "./models/".
        tokenizer (AutoTokenizer | str): The tokenizer to use for training. Default: "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_encoder (str): The pretrained encoder model to use. Default: "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_decoder (str): The pretrained decoder model to use. Default: "seyonec/ChemBERTa-zinc-base-v1".
        encoder_max_length (int): The maximum length of the encoder input. Default: 512.
        decoder_max_length (int): The maximum length of the decoder input. Default: 512.
        tie_encoder_decoder (bool): Whether to tie the encoder and decoder weights. Default: False.
        delete_repo_if_exists (bool): Whether to delete the repository if it already exists. Default: False.
        delete_local_repo_if_exists (bool): Whether to delete the local repository if it already exists. Default: False.
        training_args (Dict[str, Any]): The training arguments for the Trainer. Default: None.
        resume_from_checkpoint (str): The checkpoint to resume training from. Default: None.
        num_optuna_trials (int): The number of Optuna hyperparameter search trials. Default: 0.
        num_proc_map (int): The number of processes to use for mapping. Default: 1.
        per_device_batch_size (int): The batch size per device. If defined, it will overwrite batch_size. Default: None.
        lr_scheduler_type (str): The learning rate scheduler type. Default: None.
        mlm_probability (float): The probability of masking tokens in the input. Default: 0.15.
        randomize_smiles (bool): Whether to randomize SMILES strings. Default: False.
        randomize_smiles_prob (float): The probability of randomizing SMILES strings. Default: 0.5.
        randomize_smiles_repeat (int): The number of times to repeat randomizing SMILES strings. Default: 1.
    """
        # Check if resume_from_checkpoint exists and it's a file
    if resume_from_checkpoint is not None:
        # Check if the checkpoint exists: it can be either a file or a directory
        if not Path(resume_from_checkpoint).exists():
            raise ValueError(f"Checkpoint file '{resume_from_checkpoint}' does not exist.")

    if hub_token is not None:
        hf.login(token=hub_token)
    
    # Setup output directory and Hugging Face repository
    output_dir += f"/{model_id}"
    if organization is not None:
        hub_model_id = f"{organization}/{model_id}"
        if delete_repo_if_exists and repo_exists(hub_model_id, token=hub_token):
            delete_hf_repository(repo_id=hub_model_id, token=hub_token)
            if not repo_exists(hub_model_id, token=hub_token):
                print(f"Repository '{hub_model_id}' deleted.")
            else:
                print(f"Repository '{hub_model_id}' could not be deleted.")
                return
        if delete_local_repo_if_exists and Path(output_dir).exists():
            subprocess.run(["rm", "-rf", output_dir])
            if not Path(output_dir).exists():
                print(f"Local repository '{output_dir}' deleted.")
            else:
                print(f"Local repository '{output_dir}' could not be deleted.")
                return
        repo_url = create_hf_repository(
            repo_id=hub_model_id,
            repo_type="model",
            exist_ok=True,
            private=True,
            token=hub_token,
        )
        print(f"Repository '{hub_model_id}' created at URL: {repo_url}")
    else:
        hub_model_id = None
    print(f"Hub model ID: {hub_model_id}")

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    elif tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_encoder)
    
    # Set the pad token to the end of the sequence, required for MLM training
    tokenizer.pad_token = tokenizer.eos_token

    # Load the tokenized dataset
    print("Loading tokenized dataset.")
    dataset_tokenized = load_tokenized_dataset(
        ds_name,
        ds_config,
        tokenizer,
        batch_size_tokenizer,
        encoder_max_length,
        decoder_max_length,
        token=hub_token,
        num_proc_map=num_proc_map,
        randomize_smiles=randomize_smiles,
        randomize_smiles_prob=randomize_smiles_prob,
        randomize_smiles_repeat=randomize_smiles_repeat,
        randomize_text=True,
        randomize_labels=False,
    )
    # Remove "labels" column from the dataset
    dataset_tokenized = dataset_tokenized.remove_columns(["labels"])
    print("Dataset loaded.")

    # Setup the model for `model_init` in the Trainer
    bert2bert = lambda: get_encoder_decoder_model(
        pretrained_encoder=pretrained_encoder,
        pretrained_decoder=pretrained_decoder,
        max_length=encoder_max_length,
        tie_encoder_decoder=tie_encoder_decoder,
    )

    # Setup the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=8,
    )

    # Setup the training arguments
    if per_device_batch_size is None:
        per_device_batch_size = batch_size // gradient_accumulation_steps
    if training_args is None:
        training_args = {
            "output_dir": output_dir,
            # Optimizer-related configs
            "learning_rate": learning_rate,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine" if lr_scheduler_type is None else lr_scheduler_type,
            "warmup_steps": 8000, # NOTE: ChemFormer: 8000
            # "warmup_ratio": 0,
            "adam_beta1": 0.9, # NOTE: ChemFormer: 0.9
            "adam_beta2": 0.999, # NOTE: ChemFormer: 0.999
            "adam_epsilon": 1e-8, # Default: 1e-8
            # Batch size, device, and performance optimizations configs
            # "torch_compile": True,
            "group_by_length": True,
            "per_device_train_batch_size": per_device_batch_size,
            "per_device_eval_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "auto_find_batch_size": True,
            "fp16": True if torch.cuda.is_available() else False,
            # Evaluation and checkpointing configs
            "max_steps": max_steps,
            "num_train_epochs": num_train_epochs,
            "save_steps": 1000, # NOTE: 200
            "save_strategy": "steps",
            "eval_steps": 1000, # NOTE: 500
            "evaluation_strategy": "steps",
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "perplexity",
            "include_inputs_for_metrics": True,
            # Logging configs
            "log_level": "warning",
            "logging_steps": 500,
            "disable_tqdm": True,
            "report_to": ["tensorboard"],
            "save_only_model": False, # Default: False
            # Hub information configs
            "push_to_hub": True, # NOTE: Also manually done further down
            "push_to_hub_model_id": model_id,
            "push_to_hub_organization": organization,
            "hub_model_id": hub_model_id,
            "hub_token": hub_token,
            "hub_strategy": "checkpoint", # NOTE: Allows to resume training from last checkpoint 
            "hub_private_repo": True,
            # Other configs
            "seed": 42,
            "data_seed": 42,
        }

    # Setup the Trainer and start training (no Optuna hyperparameter search)
    trainer = Trainer(
        model_init=bert2bert,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=TrainingArguments(**training_args),
        compute_metrics=compute_metrics_for_mlm,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
    )
    if resume_from_checkpoint is not None:
        trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
        )
    else:
        trainer.train()
    print("-" * 80)
    print("Training completed.")
    print("-" * 80)

    if hub_model_id is not None:
        print("Pushing model to Hugging Face Hub.")
        print("-" * 80)
        tokenizer.save_pretrained(output_dir)
        trainer.push_to_hub(
            commit_message="Initial version",
            model_name=hub_model_id,
            license="mit",
            finetuned_from=f"{pretrained_encoder}",
            tasks=["Text2Text Generation", "question-answering"],
            tags=["PROTAC", "cheminformatics"],
            dataset=[ds_name],
            dataset_args=[ds_config],
        )
        tokenizer.push_to_hub(
            repo_id=hub_model_id,
            commit_message="Upload tokenizer",
            private=True,
            token=hub_token,
            tags=["PROTAC", "cheminformatics"],
        )
    print("All done.")