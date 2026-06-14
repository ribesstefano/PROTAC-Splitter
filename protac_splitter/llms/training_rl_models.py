""" Train a PPO and DPO model for PROTAC-Splitter using Hugging Face
Transformers and TRL. This is a work in progress code, so it's not tested nor
used in the package.
"""
from pathlib import Path
from typing import Optional, Literal
from functools import partial
import os
import subprocess

import torch
import evaluate
import huggingface_hub as hf
from tqdm import tqdm
from datasets import load_dataset
from rdkit import Chem
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    EncoderDecoderModel,
    AutoConfig,
)
from trl import (
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    DPOTrainer,
)

from protac_splitter.llms.data_utils import (
    load_trl_dataset,
    data_collator_for_trl,
)

from protac_splitter.llms.hf_utils import (
    create_hf_repository,
    delete_hf_repository,
    repo_exists,
)
from protac_splitter.llms.evaluation import decode_and_get_metrics
from protac_splitter.evaluation import check_substructs, split_prediction


def clean_text(text: str) -> str:
    """ Cleans the text by removing special tokens. """
    return text.replace("<s>", "").replace("</s>", "")


def reward_function(
        query: str,
        response: str,
) -> float:
    """ Reward function for the RL-based models.
    
    Args:
        query (str): The query SMILES string.
        response (str): The response SMILES string.
        
    Returns:
        float: The reward value.
    """

    substructs = split_prediction(response)
    if substructs is None:
        return torch.Tensor(-1.)
    
    if not check_substructs(
        protac_smiles=query,
        poi_smiles=substructs['poi'],
        linker_smiles=substructs['linker'],
        e3_smiles=substructs['e3'],
        return_bond_types=False,
        poi_attachment_id=1,
        e3_attachment_id=2,
    ):
        return torch.Tensor(0.)

    return torch.Tensor(1.)


def train_ppo_model(
    model_id: str = "PROTAC-Splitter-PPO-standard_rand_recombined-ChemBERTa-zinc-base",
    organization: str = 'ailab-bio',
    output_dir: str = "./models/",
    max_steps: int = 2000,
    ppo_epochs: int = 5,
    batch_size: int = 128,
    hub_token: Optional[str] = None,
    pretrained_model_name: str = "ailab-bio/PROTAC-Splitter-standard_rand_recombined-ChemBERTa-zinc-base",
    max_length: int = 512,
    delete_repo_if_exists: bool = False,
    delete_local_repo_if_exists: bool = False,
    ds_name: str = "ailab-bio/PROTAC-Splitter-Dataset",
    ds_config: str = "standard",
):
    """ Trains a PPO model on a given dataset.
    
    Args:
        model_id (str, optional): The name of the model to be trained. Defaults to "PROTAC-Splitter-PPO-standard_rand_recombined-ChemBERTa-zinc-base".
        organization (str, optional): The organization name. Defaults to 'ailab-bio'.
        output_dir (str, optional): The output directory. Defaults to "./models/".
        max_steps (int, optional): The maximum number of training steps. Defaults to 2000.
        ppo_epochs (int, optional): The number of PPO epochs. Defaults to 4.
        batch_size (int, optional): The batch size. Defaults to 128.
        hub_token (Optional[str], optional): The Hugging Face token. Defaults to None.
        pretrained_model_name (str, optional): The name of the pretrained model. Defaults to "ailab-bio/PROTAC-Splitter-standard_rand_recombined-ChemBERTa-zinc-base".
        max_length (int, optional): The maximum length of the input sequence. Defaults to 512.
        delete_repo_first (bool, optional): Whether to delete the repository first. Defaults to False.
    """
    if ppo_epochs < 1:
        raise ValueError(f"ppo_epochs must be >= 1, got {ppo_epochs}.")
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

    # Load pretrained model
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        pretrained_model_name,
        max_length=max_length,
    )
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
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
        dataset_name=ds_name,
        ds_config=ds_config,
    ).shuffle(seed=42).flatten_indices()

    # Setup PPO trainer
    hub_configs = {
        "repo_id": hub_model_id,
        "commit_message": "Initial version",
        "private": True,
    }
    ppo_config = PPOConfig(
        # Learning parameters
        learning_rate=1e-5,
        steps=max_steps, # Default: 20_000
        ppo_epochs=ppo_epochs, # Default: 4
        batch_size=batch_size, # Default: 256
        gradient_accumulation_steps=1, # Default: 1
        optimize_device_cache=True,
        # PPO parameters
        init_kl_coef=1.0,
        adap_kl_ctrl=True,
        target=0.5,
        horizon=1000,
        cliprange=0.1,
        early_stopping=True,
        target_kl=0.5,
        max_grad_norm=1.0,
        use_score_scaling=True,
        use_score_norm=True,
        whiten_rewards=True,
        # Logging parameters
        # NOTE: Check this guide for more information about the logged metrics:
        # https://huggingface.co/docs/trl/v0.10.1/logging
        model_name=hub_model_id,
        push_to_hub_if_best_kwargs=hub_configs,
        log_with="tensorboard", # ["wandb", LoggerType.TENSORBOARD],
        project_kwargs={"logging_dir": output_dir},
        seed=42,
    )
    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        num_shared_layers=0,
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=data_collator_for_trl,
        # lr_scheduler=torch.optim.lr_scheduler.LRScheduler, # NOTE: It must be that, CosineAnnealingLR is not supported
    )

    # Training Loop
    generation_kwargs = {
        "do_sample": True,
        "num_beams": 5,
        "top_k": 20,
        "max_length": 512,
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

    # Save model and tokenizer
    ppo_trainer.push_to_hub(**hub_configs)
    tokenizer.push_to_hub(**hub_configs)


def train_dpo_model(
    model_name: str = "ailab-bio/PROTAC-Splitter-DPO",
    output_dir: str = "./models/",
    beta: float = 0.1,
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
    learning_rate: float = 5e-5,
    max_steps: int = 2000,
    num_train_epochs: int = -1,
    batch_size: int = 128,
    gradient_accumulation_steps: int = 4,
    resume_from_checkpoint: bool = False,
    hub_token: Optional[str] = None,
    pretrained_model_name: str = "ailab-bio/PROTAC-Splitter_untied_80-20-split",
    pretrained_ref_model_name: str = "ailab-bio/PROTAC-Splitter_untied_80-20-split",
    max_length: int = None,
    delete_repo_first: bool = False,
    optuna_search: bool = False,  
):
    """ Trains a DPO model on a given dataset.
    
    Args:
        model_name (str, optional): The name of the model to be trained. Defaults to "ailab-bio/PROTAC-Splitter-DPO".
        max_steps (int, optional): The maximum number of training steps. Defaults to 2000.
    """
    if hub_token is not None:
        hf.login(token=hub_token)
    if delete_repo_first and not resume_from_checkpoint:
        delete_hf_repository(repo_id=model_name, token=hub_token)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name,
        token=hub_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Get train and eval datasets
    dataset = load_dataset(
        "ailab-bio/PROTAC-Substructures-DPO",
        token=hub_token,
    )
    # Setup models
    def model_init():
        return EncoderDecoderModel.from_pretrained(
            pretrained_model_name,
            token=hub_token,
        )
    model_ref = EncoderDecoderModel.from_pretrained(
        pretrained_ref_model_name,
        token=hub_token,
    )
    # Setup training arguments
    per_device_batch_size = batch_size // gradient_accumulation_steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        # Optimizer-related configs
        learning_rate=learning_rate,
        optim="adamw_torch",
        lr_scheduler_type="cosine", # Default: "linear"
        # Batch size and device configs
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        auto_find_batch_size=True,
        # torch_compile=True,
        fp16=True,
        # Evaluation and checkpointing configs
        evaluation_strategy="steps", # TODO: Why is it not working? "steps",
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        eval_steps=100,
        save_steps=200,
        # eval_steps=7500,
        # warmup_steps=2000,
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        # metric_for_best_model="valid_smiles",
        # Logging configs
        log_level="info",
        logging_steps=50,
        disable_tqdm=True,
        # Hub information configs
        push_to_hub=True, # NOTE: Done manually further down
        hub_token=hub_token,
        hub_model_id=model_name,
        hub_strategy="checkpoint", # NOTE: Allows to resume training from last checkpoint 
        hub_private_repo=True,
        # Other configs
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
    )
    # Setup Matrics
    # TODO: The metric is not working because the predictions include rewards,
    # or something like that, i.e., real values, which cannot be decoded by the
    # tokenizer. Skipping for now and using the default one.
    rouge = evaluate.load("rouge")
    fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=8,
        fpSize=2048,
    )
    metric = partial(
        decode_and_get_metrics,
        rouge=rouge,
        tokenizer=tokenizer,
        fpgen=fpgen,
    )
    # Setup trainer and start training
    if max_length is None:
        max_length = AutoConfig.from_pretrained(
            pretrained_model_name,
            token=hub_token,
        ).max_length
        # max_length = model.config.max_length
    dpo_trainer = DPOTrainer(
        model=model_init(),
        ref_model=model_ref,
        beta=beta,
        loss_type=loss_type,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        model_init=model_init if optuna_search else None,
        # compute_metrics=metric,
        max_length=max_length,
        max_prompt_length=max_length,
        max_target_length=max_length,
        is_encoder_decoder=True,
        padding_value=tokenizer.pad_token_id,
        truncation_mode="keep_start",
        args=training_args,
    )
    if optuna_search and False:
        # TODO: This is not working because the training arguments do NOT
        # include the beta parameter...
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
                "beta": trial.suggest_float("beta", 0.1, 0.5),
            }
        best_trials = dpo_trainer.hyperparameter_search(
            direction=["minimize"],
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=20,
            # compute_objective=compute_objective,
        )
        print("-" * 80)
        print(f"Best trials:\n{best_trials}")
        print("-" * 80)
    else:
        if resume_from_checkpoint:
            resume_from_checkpoint = "last-checkpoint"
        else:
            resume_from_checkpoint = None
        dpo_trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
        )
    dpo_trainer.push_to_hub(
        commit_message="Initial version",
        model_name=model_name,
        license="mit",
        finetuned_from=pretrained_model_name,
        tasks=["Text2Text Generation"],
        tags=["PROTAC", "cheminformatics"],
        dataset="ailab-bio/PROTAC-Substructures-DPO",
    )