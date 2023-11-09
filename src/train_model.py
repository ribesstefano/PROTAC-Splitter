import os
import evaluate
import math
import torch
import huggingface_hub as hf
from model_utils import get_model
from data_utils import (
    load_tokenized_dataset,
    load_trl_dataset,
    data_collator_for_trl,
)
from evaluation_metrics import compute_metrics_with_chem, reward_function
from tqdm import tqdm
from datasets import load_dataset
from typing import Optional
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
)
from trl import (
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)


def create_hf_repository(**kwargs):
  """Creates a new Hugging Face repository."""
  api = hf.HfApi()
  return api.create_repo(**kwargs)


def delete_hf_repository(**kwargs):
  """Creates a new Hugging Face repository."""
  print(f'Deleting repository {kwargs["repo_id"]}.')
  api = hf.HfApi()
  return api.delete_repo(**kwargs)


def train_model(
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
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_encoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_decoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    encoder_max_length: int = 512,
    decoder_max_length: int = 512,
    tie_encoder_decoder: bool = False,
    delete_repo_first: bool = False,
    training_args: Optional[Seq2SeqTrainingArguments] = None,
):
    """Trains a model on a given dataset.
    
    Args:
        model_id (str): The name of the model to be trained.
        ds_name (str): The name of the dataset to be used for training.
        ds_config (str, optional): The name of the dataset configuration to be used for training. Defaults to 'default'.
        learning_rate (float, optional): The learning rate. Defaults to 5e-5.
        max_steps (int, optional): The maximum number of training steps. Defaults to -1.
        num_train_epochs (int, optional): The number of training epochs. Defaults to 40.
        batch_size (int, optional): The batch size. Defaults to 128.
        batch_size_tokenizer (int, optional): The batch size for the tokenizer. Defaults to 512.
        gradient_accumulation_steps (int, optional): The number of gradient accumulation steps. Defaults to 4.
        hub_token (Optional[str], optional): The Hugging Face token. Defaults to None.
        organization (Optional[str], optional): The Hugging Face organization. Defaults to None.
        output_dir (str, optional): The output directory. Defaults to "./models/".
        tokenizer (AutoTokenizer | str, optional): The tokenizer. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_encoder (str, optional): The name of the pretrained encoder. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_decoder (str, optional): The name of the pretrained decoder. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        encoder_max_length (int, optional): The maximum length of the encoder. Defaults to 256.
        decoder_max_length (int, optional): The maximum length of the decoder. Defaults to 256.
        delete_repo_first (bool, optional): Whether to delete the repository first. Defaults to False.
    """
    if hub_token is not None:
        hf.login(token=hub_token)
    # Disable RDKit logging: when checking SMILES validity, we suppress warnings
    RDLogger.DisableLog("rdApp.*")
    # Setup output directory and Hugging Face repository
    output_dir += f"/{model_id}"
    if organization is not None:
        hub_model_id = f"{organization}/{model_id}"
        if delete_repo_first:
            delete_hf_repository(repo_id=hub_model_id, token=hub_token)
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
    # try:
    #     bert2bert = EncoderDecoderModel.from_pretrained(hub_model_id)
    #     print(f"Skipping pretrained model {hub_model_id}.")
    # except:
    #     print('-' * 80)
    #     print(f"Training model {hub_model_id} on dataset: {ds_name}.")
    #     print('-' * 80)
    bert2bert = get_model(
        pretrained_encoder=pretrained_encoder,
        pretrained_decoder=pretrained_decoder,
        max_length=encoder_max_length,
        tie_encoder_decoder=tie_encoder_decoder,
    )
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    elif tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_encoder)
    dataset_tokenized = load_tokenized_dataset(
        ds_name,
        ds_config,
        tokenizer,
        batch_size_tokenizer,
        encoder_max_length,
        decoder_max_length,
        token=hub_token,
    )
    per_device_batch_size = batch_size // gradient_accumulation_steps
    if training_args is None:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            # Optimizer-related configs
            learning_rate=learning_rate,
            optim="adamw_torch",
            lr_scheduler_type="cosine", # Default: "linear"
            # Generation configs
            predict_with_generate=True,
            generation_num_beams=1, # Greedy strategy
            # Batch size and device configs
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            auto_find_batch_size=True,
            # torch_compile=True,
            fp16=True,
            # Evaluation and checkpointing configs
            evaluation_strategy="steps",
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            eval_steps=100,
            save_steps=200,
            # eval_steps=7500,
            # warmup_steps=2000,
            save_strategy="steps",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="valid_smiles",
            # Logging configs
            log_level="info",
            logging_steps=50,
            disable_tqdm=True,
            # Hub information configs
            push_to_hub=True, # NOTE: Done manually further down
            hub_token=hub_token,
            hub_model_id=hub_model_id,
            hub_strategy="checkpoint", # NOTE: Allows to resume training from last checkpoint 
            hub_private_repo=True,
            # Other configs
            seed=42,
            data_seed=42,
        )
    rouge = evaluate.load("rouge")
    fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=8, fpSize=2048)
    metric = partial(
        compute_metrics_with_chem,
        rouge=rouge,
        tokenizer=tokenizer,
        fpgen=fpgen,
    )
    trainer = Seq2SeqTrainer(
        model=bert2bert,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=metric,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"], # .select(range(10)),
    )
    trainer.train(
        # resume_from_checkpoint="last-checkpoint",
    )
    if hub_model_id is not None:
        trainer.push_to_hub(
            commit_message="Initial version",
            model_name=hub_model_id,
            license="mit",
            finetuned_from=f"{pretrained_encoder}",
            tasks=["Text2Text Generation"],
            tags=["PROTAC", "cheminformatics"],
            dataset=ds_name,
            dataset_args=ds_config,
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


def clean_text(text: str) -> str:
    return text.replace("<s>", "").replace("</s>", "")


def train_ppo_model(
    model_name: str = "ailab-bio/PROTAC-Splitter-PPO",
    max_steps: int = 2000,
    ppo_epochs: int = 4,
    batch_size: int = 128,
    hub_token: Optional[str] = None,
    pretrained_model_name: str = "ailab-bio/PROTAC-Splitter_untied_80-20-split",
    max_length: int = 512,
    delete_repo_first: bool = False,     
):
    """ Trains a PPO model on a given dataset.
    
    Args:
        model_name (str, optional): The name of the model to be trained. Defaults to "ailab-bio/PROTAC-Splitter-PPO".
        max_steps (int, optional): The maximum number of training steps. Defaults to 2000.
        ppo_epochs (int, optional): The number of PPO epochs. Defaults to 4.
        batch_size (int, optional): The batch size. Defaults to 128.
        hub_token (Optional[str], optional): The Hugging Face token. Defaults to None.
        pretrained_model_name (str, optional): The name of the pretrained model. Defaults to "ailab-bio/PROTAC-Splitter_untied_80-20-split".
        max_length (int, optional): The maximum length of the input sequence. Defaults to 512.
        delete_repo_first (bool, optional): Whether to delete the repository first. Defaults to False.
    """
    if ppo_epochs < 1:
        raise ValueError(f"ppo_epochs must be >= 1, got {ppo_epochs}.")
    # Disable RDKit logging: when checking SMILES validity, we suppress warnings
    RDLogger.DisableLog("rdApp.*")
    if hub_token is not None:
        hf.login(token=hub_token)
    if delete_repo_first:
        delete_hf_repository(repo_id=model_name, token=hub_token)
    # Load pretrained model
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        pretrained_model_name,
        max_length=max_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Get dataset
    train_dataset = load_trl_dataset(
        tokenizer=tokenizer,
        token=hub_token,
        max_length=max_length,
    ).shuffle(seed=42).flatten_indices()
    # Setup PPO trainer
    hub_configs = {
        "repo_id": model_name,
        "commit_message": "Initial version",
        "private": True,
    }
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        steps=max_steps, # Default: 20_000
        ppo_epochs=ppo_epochs, # Default: 4
        batch_size=batch_size, # Default: 256
        gradient_accumulation_steps=1, # Default: 1
        # global_batch_size=8,
        optimize_device_cache=True,
        push_to_hub_if_best_kwargs=hub_configs,
        seed=42,
    )
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=data_collator_for_trl,
    )
    # Training Loop
    generation_kwargs = {
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        # Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        # Compute reward score
        rewards = [reward_function(clean_text(q), clean_text(r)) for q, r in zip(batch["query"], batch["response"])]
        rewards = [torch.tensor(r) for r in rewards]
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    # Save model
    ppo_trainer.push_to_hub(**hub_configs)