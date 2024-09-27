import os
from typing import Optional, Dict, Any
from functools import partial
import subprocess

import evaluate
import huggingface_hub as hf
from rdkit import Chem
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    GenerationConfig,
)
from datasets import load_dataset

from .data_utils import load_tokenized_dataset
from .evaluation import decode_and_get_metrics
from .hf_utils import (
    create_hf_repository,
    delete_hf_repository,
    repo_exists,
)
from .model_utils import get_model


def get_lr_scheduler_kwargs(lr_scheduler_type: str) -> Dict[str, Any]:
    """ Returns the default learning rate scheduler kwargs for a given type.
    
    Args:
        lr_scheduler_type (str): The type of the learning rate scheduler.

    Returns:
        Dict[str, Any]: The default learning rate scheduler kwargs.
    """
    if lr_scheduler_type == "cosine":
        return {}
    elif lr_scheduler_type == "cosine_with_restarts":
        return {"num_cycles": 5}
    elif lr_scheduler_type == "cosine_with_min_lr":
        return {}
    elif lr_scheduler_type == "polynomial":
        return {"power": 1.0}
    elif lr_scheduler_type == "reduce_lr_on_plateau":
        return {"min_lr": 1e-6}
    else:
        raise ValueError(f"Unknown learning rate scheduler type: '{lr_scheduler_type}'")


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
    delete_repo_if_exists: bool = False,
    delete_local_repo_if_exists: bool = False,
    training_args: Optional[Dict[str, Any]] = None,
    resume_from_checkpoint: Optional[str] = None,
    num_optuna_trials: int = 0,
    num_proc_map: int = 1,
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
        delete_repo_if_exists (bool, optional): Whether to delete the repository first. Defaults to False.
        training_args (Optional[Seq2SeqTrainingArguments], optional): The training arguments. Defaults to None.
        resume_from_checkpoint (Optional[str], optional): The checkpoint to resume training from. Defaults to None.
        num_optuna_trials (int, optional): The number of Optuna trials. Defaults to 0, i.e., no Optuna hyperparameter search.
    """
    # Check if resume_from_checkpoint exists and it's a file
    if resume_from_checkpoint is not None:
        if not os.path.isfile(resume_from_checkpoint):
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
        if delete_local_repo_if_exists and os.path.exists(output_dir):
            subprocess.run(["rm", "-rf", output_dir])
            if not os.path.exists(output_dir):
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
    )
    # Precompute a "length" column for the dataset using the map function
    def add_length(x):
        x["length"] = len(x["input_ids"])
        return x
    dataset_tokenized = dataset_tokenized.map(
        add_length,
        num_proc=num_proc_map,
    )
    print("Dataset loaded.")

    # Setup the model for `model_init` in the Trainer
    bert2bert = lambda: get_model(
        pretrained_encoder=pretrained_encoder,
        pretrained_decoder=pretrained_decoder,
        max_length=encoder_max_length,
        tie_encoder_decoder=tie_encoder_decoder,
    )

    # Setup the data collator, which will efficiently pad the inputs and targets
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=bert2bert(),
        # max_length=encoder_max_length,
        pad_to_multiple_of=8,
    )

    # Setup the metric function
    rouge = evaluate.load("rouge")
    fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=11,
        fpSize=1024,
    )
    metric = partial(
        decode_and_get_metrics,
        rouge=rouge,
        tokenizer=tokenizer,
        fpgen=fpgen,
    )

    # Setup the training arguments
    per_device_batch_size = batch_size // gradient_accumulation_steps
    if training_args is None:
        generation_config = GenerationConfig(
            max_length=512,
            max_new_tokens=512,
            do_sample=True,
            num_beams=5,
            temperature=1.0,
        )
        training_args = {
            "output_dir": output_dir,
            # Optimizer-related configs
            "learning_rate": learning_rate,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "warmup_steps": 8000, # NOTE: ChemFormer: 8000
            "adam_beta1": 0.9, # NOTE: ChemFormer: 0.9
            "adam_beta2": 0.999, # NOTE: ChemFormer: 0.999
            "adam_epsilon": 1e-8, # Default: 1e-8
            # Generation configs
            "predict_with_generate": True,
            "generation_config": generation_config,
            "generation_max_length": 512,
            # Batch size, device, and performance optimizations configs
            # "torch_compile": True,
            "group_by_length": True,
            "per_device_train_batch_size": per_device_batch_size,
            "per_device_eval_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "auto_find_batch_size": True,
            "fp16": True,
            # Evaluation and checkpointing configs
            "max_steps": max_steps,
            "num_train_epochs": num_train_epochs,
            "save_steps": 1000, # NOTE: 200
            "save_strategy": "steps",
            "eval_steps": 500, # NOTE: 500
            "evaluation_strategy": "steps",
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "reassembly",
            "include_inputs_for_metrics": True,
            # Logging configs
            "log_level": "warning",
            "logging_steps": 500,
            "disable_tqdm": True,
            "report_to": ["tensorboard"],
            "save_only_model": False, # Default: False
            # Hub information configs
            "push_to_hub": True, # NOTE: Also manually done further down
            "hub_token": hub_token,
            "hub_model_id": hub_model_id,
            "hub_strategy": "checkpoint", # NOTE: Allows to resume training from last checkpoint 
            "hub_private_repo": True,
            # Other configs
            "seed": 42,
            "data_seed": 42,
        }

    # Modify the training arguments with Optuna hyperparameter search
    if num_optuna_trials > 0:
        def optuna_hp_space(trial):

            # NOTE: Tuning generation config is not implemented yet, please refer to this issue: https://github.com/huggingface/transformers/issues/33755
            # ------------------------------------------------------------------
            # # Define default generation parameters
            # generation_params = {
            #     "max_length": 512,
            #     "max_new_tokens": 512,
            #     'top_k': 20,
            # }
            # 
            # # Define the generation strategies and pick one with Optuna
            # # REF: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/generation/configuration_utils.py#L71
            # generation_strategy_params = {
            #     "greedy": {"num_beams": 1, "do_sample": False},
            #     "contrastive_search": {"penalty_alpha": 0.1, "top_k": 10},
            #     "multinomial_sampling": {"num_beams": 1, "do_sample": True},
            #     "beam_search_decoding": {"num_beams": 5, "do_sample": False},
            #     "beam_search_multinomial_sampling": {"num_beams": 5, "do_sample": True},
            #     "diverse_beam_search_decoding": {"num_beams": 5, "num_beam_groups": 5, "diversity_penalty": 1.0},
            # }
            # gen_strategy = trial.suggest_categorical("generation_strategy", list(generation_strategy_params.keys()))
            # generation_params.update(generation_strategy_params[gen_strategy])
            # 
            # # Update the generation params with the temperature
            # temperature = trial.suggest_float("temperature", 0.5, 1.1, step=0.1)
            # generation_params["temperature"] = temperature
            # 
            # # # Instantiate a GenerationConfig object to pass to the Trainer arguments
            # # generation_config = GenerationConfig(**generation_params)
            # ------------------------------------------------------------------

            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
            lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["cosine", "cosine_with_restarts", "reduce_lr_on_plateau"]) # "cosine_with_min_lr", "polynomial"
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1, step=0.01)
            generation_num_beams = trial.suggest_categorical("generation_num_beams", [1, 5])

            # Change the number of evaluation steps based on the warmup ratio
            # suggested by Optuna. This way, we should not prune too early
            num_steps = max(max_steps, num_train_epochs * len(dataset_tokenized["train"]) // batch_size)
            warmup_steps = int(warmup_ratio * num_steps)
            hp_eval_steps = warmup_steps * 2

            return {
                "learning_rate": learning_rate,
                "lr_scheduler_type": lr_scheduler_type,
                "lr_scheduler_kwargs": get_lr_scheduler_kwargs(lr_scheduler_type),
                "warmup_ratio": warmup_ratio,
                "generation_num_beams": generation_num_beams,
                "eval_steps": hp_eval_steps,
                # "generation_config": generation_config,
                # "generation_config": generation_params,
                # **{f"generation_{k}": v for k, v in generation_params.items()},
            }

        def compute_objective(metrics: Dict[str, float]):
            # NOTE: Having a higher eval_reassembly score should also correspond
            # to a low eval loss, so we just focus on the reassembly score.
            return metrics["eval_reassembly"]

        # REF: Evaluate a bit more often than the default to be able to prune
        # bad trials early.
        # NOTE: Since the warmup can affect the objective, we instead increase
        # the eval_steps to prevent pruning too early when the LR is still
        # warming up.
        num_steps = max(max_steps, num_train_epochs * len(dataset_tokenized["train"]) // batch_size)
        eval_steps = training_args.get("eval_steps", 500)
        hp_eval_steps = eval_steps * 2
        training_args["eval_steps"] = hp_eval_steps + num_steps % hp_eval_steps

        # Setup a "fake" Trainer for the hyperparameter search
        trainer = Seq2SeqTrainer(
            model_init=bert2bert,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=Seq2SeqTrainingArguments(**(training_args.copy())),
            compute_metrics=metric,
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["test"],
        )
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=num_optuna_trials,
            compute_objective=compute_objective,
        )

        # Set the best hyperparameters in the Trainer arguments (and log them)
        print("-" * 80)
        print(f"Best trial objective: {best_trial.objective:.4f} (eval_reassembly)")
        for hparam, value in best_trial.hyperparameters.items():
            print(f"\t* {hparam}: {value}")
            training_args[hparam] = value
            if hparam == "lr_scheduler_type":
                training_args["lr_scheduler_kwargs"] = get_lr_scheduler_kwargs(value)
        print("-" * 80)

        # Setup the original eval_steps
        training_args["eval_steps"] = eval_steps
        
    # Setup the Trainer and start training (with best hyperparameters)
    trainer = Seq2SeqTrainer(
        model_init=bert2bert,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=Seq2SeqTrainingArguments(**training_args),
        compute_metrics=metric,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
    )
    if resume_from_checkpoint is not None and num_optuna_trials > 0:
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
