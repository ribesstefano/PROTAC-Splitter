import os
from typing import Optional, Dict, Any, Callable, Tuple, Union
from functools import partial
import subprocess
import copy
import datetime
import logging
import math
import json

import torch
import numpy as np
import huggingface_hub as hf
from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    GenerationConfig,
    TrainerCallback,
    set_seed,
)
from accelerate.utils import write_basic_config
from accelerate import Accelerator

import optuna
from optuna.samplers import QMCSampler
from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    ThresholdPruner,
    PatientPruner,
    MedianPruner,
)
from optuna.study._study_direction import StudyDirection

from .data_utils import load_tokenized_dataset
from .evaluation import decode_and_get_metrics
from .hf_utils import (
    create_hf_repository,
    delete_hf_repository,
    repo_exists,
    upload_single_file,
)
from .model_utils import get_encoder_decoder_model, get_causal_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU with index 0
# logging.basicConfig(level=logging.DEBUG)

class PrintStepCallback(TrainerCallback):

    def on_init_end(self, args, state, control, **kwargs):
        print(f"[{datetime.datetime.now()}] Initialization complete. Training is starting.")

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            print(f"[{datetime.datetime.now()}] Global step: {state.global_step:,}")


class ScoreMetric:

    def __init__(self):
        self.batch_scores = []
    
    def update(self, scores):
        self.batch_scores.append(scores)
    
    def compute(self):
        all_labels = set()
        for scores in self.batch_scores:
            all_labels.update(scores.keys())
        
        aggregate_scores = {}
        for k in all_labels:
            scores = [s.get(k, np.nan) for s in self.batch_scores]
            print(f"{k}: {np.nanmean(scores):.4f}")
            aggregate_scores[k] = np.nanmean(scores)
        
        self.batch_scores = []
        return aggregate_scores


score_metric = ScoreMetric()
hp_score_metric = ScoreMetric()


class WrappedEarlyStoppingPruner(BasePruner):
    """
    Pruner that wraps another pruner and checks if the trial should be pruned.
    It first evaluates the wrapped pruner and, if the wrapped pruner suggests
    pruning, prune. Otherwise, evaluates based on a patience threshold with a
    tolerance (min_delta) and eventually prunes.
    
    Args:
        wrapped_pruner:
            Wrapped pruner to check first. Pruning is only applied if this pruner recommends it.
        patience:
            Number of steps to wait for an improvement before pruning.
        min_delta:
            Minimum improvement required to reset patience.
        n_warmup_steps:
            Number of initial steps to skip the patience check.
    """
    
    def __init__(
            self, 
            wrapped_pruner: BasePruner,
            patience: int, 
            min_delta: float = 0.0,
            n_warmup_steps: int = 0,
    ) -> None:
        if wrapped_pruner is None or not isinstance(wrapped_pruner, BasePruner):
            raise ValueError(f"wrapped_pruner must be an instance of BasePruner but got {wrapped_pruner}.")
        if patience < 0:
            raise ValueError(f"patience cannot be negative but got {patience}.")
        if min_delta < 0:
            raise ValueError(f"min_delta cannot be negative but got {min_delta}.")
        if n_warmup_steps < 0:
            raise ValueError(f"n_warmup_steps cannot be negative but got {n_warmup_steps}.")

        self._wrapped_pruner = wrapped_pruner
        self._patience = patience
        self._min_delta = min_delta
        self._n_warmup_steps = n_warmup_steps

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        step = trial.last_step
        if step is None:
            return False

        intermediate_values = trial.intermediate_values
        steps = np.asarray(list(intermediate_values.keys()))

        # If there are insufficient steps or we are still in the warmup phase, do not prune.
        if steps.size <= self._patience + 1 or step < self._n_warmup_steps:
            return False

        # First, check the wrapped pruner. If it suggests pruning, prune.
        if self._wrapped_pruner.prune(study, trial):
            return True
        
        steps.sort()

        # This is the score patience steps ago
        steps_before_patience = steps[: -self._patience - 1]
        scores_before_patience = np.asarray(
            list(intermediate_values[step] for step in steps_before_patience)
        )

        # And these are the scores after that
        steps_after_patience = steps[-self._patience - 1 :]
        scores_after_patience = np.asarray(
            list(intermediate_values[step] for step in steps_after_patience)
        )

        direction = study.direction
        if direction == StudyDirection.MINIMIZE:
            should_prune = np.nanmin(scores_before_patience) + self._min_delta < np.nanmin(
                scores_after_patience
            )
        else:
            should_prune = np.nanmax(scores_before_patience) - self._min_delta > np.nanmax(
                scores_after_patience
            )

        return should_prune


def get_lr_scheduler_kwargs(lr_scheduler_type: str) -> Dict[str, Any]:
    """ Returns the default learning rate scheduler kwargs for a given type.

    Reference: https://huggingface.co/docs/timm/en/reference/schedulers
    
    Args:
        lr_scheduler_type (str): The type of the learning rate scheduler.

    Returns:
        Dict[str, Any]: The default learning rate scheduler kwargs.
    """
    if lr_scheduler_type == "cosine":
        return {}
    elif lr_scheduler_type == "cosine_with_restarts":
        return {"num_cycles": 3}
    elif lr_scheduler_type == "cosine_with_min_lr":
        return {}
    elif lr_scheduler_type == "polynomial":
        return {"power": 1.0}
    elif lr_scheduler_type == "reduce_lr_on_plateau":
        return {"min_lr": 1e-6}
    else:
        raise ValueError(f"Unknown learning rate scheduler type: '{lr_scheduler_type}'")


def get_best_hyperparameters(
        model_init: Callable,
        tokenizer: AutoTokenizer,
        data_collator: Union[DataCollatorForSeq2Seq, DataCollatorForLanguageModeling],
        compute_metrics: Callable,
        dataset_tokenized: Dict[str, Any],
        training_args: Dict[str, Any],
        num_optuna_trials: int,
        lr_scheduler_type: Optional[str] = None,
        causal_language_modeling: bool = False,
        all_fragments_as_labels: bool = True,
        linkers_only_as_labels: bool = False,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Runs an Optuna hyperparameter search to find the best hyperparameters.

    Args:
        model_init (Callable): The model initialization function.
        tokenizer (AutoTokenizer): The tokenizer.
        data_collator (DataCollatorForSeq2Seq): The data collator.
        compute_metrics (Callable): The compute metrics function.
        dataset_tokenized (Dict[str, Any]): The tokenized dataset.
        training_args (Dict[str, Any]): The training arguments.
        num_optuna_trials (int): The number of Optuna trials.

    Returns:
        Tuple[float, Dict[str, Any], Dict[str, Any]]: The best objective, the best hyperparameters, and the best training arguments.
    """
    def optuna_hp_space(trial):
        # NOTE: Tuning generation config is not implemented yet, please refer to this issue: https://github.com/huggingface/transformers/issues/33755
        # Suggest hparams "shared" across all scheduler types
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.1, step=0.01)

        # NOTE: We might want to use QMCSampler instead of TPESampler, which
        # doesn't support categorical parameters. Categories can be encoded as
        # integers and then decoded back to the original categories.

        # NOTE: According to the GitHub code, the number of training and warmup
        # steps for the scheduler types are automatically set, we don't need to
        # pass them in the lr_scheduler_kwargs.

        if lr_scheduler_type is None:
            lr_scheduler_types = ["cosine", "cosine_with_restarts", "reduce_lr_on_plateau"] # "cosine_with_min_lr", "polynomial"
            suggested_lr_sched = trial.suggest_int("lr_scheduler_type", 0, len(lr_scheduler_types) - 1)
            suggested_lr_sched = lr_scheduler_types[suggested_lr_sched]
            lr_scheduler_kwargs = get_lr_scheduler_kwargs(lr_scheduler_type)
        elif lr_scheduler_type == "cosine":
            lr_scheduler_kwargs = {
                "num_cycles": trial.suggest_float("num_cycles", 0.5, 10, step=0.5),
            }
        elif lr_scheduler_type == "cosine_with_restarts":
            lr_scheduler_kwargs = {
                "num_cycles": trial.suggest_int("num_cycles", 1, 10, step=1),
            }
        elif lr_scheduler_type == "reduce_lr_on_plateau":
            lr_scheduler_kwargs = {
                "min_lr": trial.suggest_float("min_lr", 1e-12, 1e-9, log=True),
                "factor": trial.suggest_float("factor", 0.1, 0.99, step=0.01),
            }

        return {
            "lr_scheduler_kwargs": lr_scheduler_kwargs,
            "lr_scheduler_type": lr_scheduler_type if lr_scheduler_type is not None else suggested_lr_sched,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
        }

    if causal_language_modeling:
        def compute_objective(metrics: Dict[str, float]):
            # NOTE: We want to minimize the model perplexity, which is the
            # exponential of the negative log-likelihood loss. Optuna is setup
            # to maximize the objective, so we return the negative perplexity.
            return -math.exp(metrics["eval_loss"])
    else:
        if all_fragments_as_labels:
            def compute_objective(metrics: Dict[str, float]):
                # NOTE: Having a higher eval_reassembly score should also correspond
                # to a low eval loss, so we just focus on the reassembly score.
                return metrics["eval_all_ligands_equal"]
        else:
            if linkers_only_as_labels:
                def compute_objective(metrics: Dict[str, float]):
                    return metrics["eval_linker_equal"]
            else:
                def compute_objective(metrics: Dict[str, float]):
                    return metrics["eval_e3_equal"] + metrics["eval_poi_equal"]
    
    def hp_name(trial: Any) -> str:
        trial_name = f"trial-number={trial.number}"
        for hparam, value in trial.params.items():
            # Check if the value is a float and round it to 3 decimals
            if hparam == "learning_rate":
                value = f"{value:.1e}"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            trial_name += f"-{hparam}={value}"
        return trial_name

    # Override the training steps
    hp_training_args = copy.deepcopy(training_args)
    hp_training_args["num_train_epochs"] = -1
    hp_training_args["max_steps"] = 10_000
    hp_training_args["eval_steps"] = 2500
    hp_training_args["eval_delay"] = 5000 # TODO: Double check if this is needed
    hp_training_args["logging_steps"] = 500
    hp_training_args["save_steps"] = 5000
    if not causal_language_modeling:
        # Use greedy decoding for the evaluation during HP search
        hp_training_args["generation_config"] = GenerationConfig(
            max_length=512,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
        )

    print("Hyperparameter search training arguments:")
    for k, v in hp_training_args.items():
        if 'token' in k:
            continue
        print(f"  - {k}: {v}")
        
    if causal_language_modeling:
        TrainerClass = Trainer
        TrainingArgumentsClass = TrainingArguments
    else:
        TrainerClass = Seq2SeqTrainer
        TrainingArgumentsClass = Seq2SeqTrainingArguments

    # Setup a "fake" Trainer for the hyperparameter search
    trainer = TrainerClass(
        model_init=model_init,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=TrainingArgumentsClass(**hp_training_args),
        compute_metrics=compute_metrics,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
        callbacks=[PrintStepCallback],
    )

    # Setup the Optuna pruner and sampler
    max_warmup_ratio = 0.1
    pruner = WrappedEarlyStoppingPruner(
        MedianPruner(
            n_startup_trials=0,
            interval_steps=1,
            n_warmup_steps=int(max_warmup_ratio * hp_training_args["max_steps"]),
        ),
        patience=5, # Check every 5000 training steps
        min_delta=0.01,
        n_warmup_steps=int(max_warmup_ratio * hp_training_args["max_steps"]),
    )
    sampler = QMCSampler(scramble=True, seed=42)

    # NOTE: The Trainer will return a BestRun object, not the Optuna trial
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        hp_name=hp_name,
        n_trials=num_optuna_trials,
        compute_objective=compute_objective, # Default: Will sum over all metrics but loss
        sampler=sampler,
        pruner=pruner,
    )

    # Set the best hyperparameters in the original Trainer arguments
    try:
        print("-" * 80)
        print(f"Best trial objective: {best_run.objective:.4f}. Summary: {best_run.run_summary}")
    except Exception as e:
        print(e)
        print("WARNING. Best trial objective could not be printed.")

    return best_run, hp_training_args


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
        num_optuna_trials: int = 0,
        num_proc_map: int = 1,
        per_device_train_batch_size: Optional[int] = None,
        per_device_eval_batch_size: Optional[int] = None,
        lr_scheduler_type: Optional[str] = None,
        cache_dir: Optional[str] = None,
        randomize_smiles: bool = False,
        randomize_smiles_prob: float = 0.0,
        all_fragments_as_labels: bool = True,
        linkers_only_as_labels: bool = False,
        warmup_ratio: Optional[float] = None,
        num_cycles: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        causal_language_modeling: bool = False,
        train_size_ratio: float = 1.0,
        training_args_bin: Optional[str] = None,
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
    set_seed(42)

    # if torch.cuda.is_available():
    #     write_basic_config(mixed_precision='fp16')
    accelerator = Accelerator()
    accelerator.print(f"Accelerator state from the current environment:\n{accelerator.state}")

    # Check if resume_from_checkpoint exists and it's a file
    if resume_from_checkpoint is not None:
        # Check if the checkpoint exists: it can be either a file or a directory
        if not os.path.exists(resume_from_checkpoint):
            raise ValueError(f"Checkpoint file '{resume_from_checkpoint}' does not exist.")

    if hub_token is not None:
        hf.login(token=hub_token)
    
    # Setup output directory and Hugging Face repository
    output_dir += f"/{model_id}"
    if organization is not None:
        hub_model_id = f"{organization}/{model_id}"
        if delete_local_repo_if_exists and os.path.exists(output_dir):
            subprocess.run(["rm", "-rf", output_dir])
            if not os.path.exists(output_dir):
                print(f"Local repository '{output_dir}' deleted.")
            else:
                print(f"Local repository '{output_dir}' could not be deleted.")
                return
        if delete_repo_if_exists and repo_exists(hub_model_id, token=hub_token):
            delete_hf_repository(repo_id=hub_model_id, token=hub_token, missing_ok=True)
            print(f"Repository '{hub_model_id}' deleted.")

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
        cache_dir=cache_dir,
        randomize_smiles=randomize_smiles,
        randomize_smiles_prob=randomize_smiles_prob,
        all_fragments_as_labels=all_fragments_as_labels,
        linkers_only_as_labels=linkers_only_as_labels,
        causal_language_modeling=causal_language_modeling,
        train_size_ratio=train_size_ratio,
    )
    print("Dataset loaded.")

    if causal_language_modeling:
        # Setup the model for `model_init` in the Trainer
        model_lambda = lambda: get_causal_model(
            pretrained_model=pretrained_decoder,
        )

        # Setup the data collator, which will efficiently pad the inputs and targets
        data_collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
            pad_to_multiple_of=8, # Default: None, Original: 8
        )
    else:
        # Precompute a "length" column for the dataset using the map function
        def add_length(x):
            x["length"] = len(x["input_ids"])
            return x
        dataset_tokenized = dataset_tokenized.map(
            add_length,
            num_proc=num_proc_map,
        )

        # Setup the model for `model_init` in the Trainer
        model_lambda = lambda: get_encoder_decoder_model(
            pretrained_encoder=pretrained_encoder,
            pretrained_decoder=pretrained_decoder,
            max_length=encoder_max_length,
            tie_encoder_decoder=tie_encoder_decoder,
        )

        # Setup the data collator, which will efficiently pad the inputs and targets
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model_lambda(),
            pad_to_multiple_of=32, # Default: None, Original: 8
        )

    # Setup the training arguments
    if per_device_train_batch_size is None:
        per_device_train_batch_size = batch_size // gradient_accumulation_steps
    if per_device_eval_batch_size is None:
        per_device_eval_batch_size = batch_size // gradient_accumulation_steps
    if training_args is None:
        training_args = {
            "output_dir": output_dir,
            # Optimizer-related configs
            "learning_rate": learning_rate,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine" if lr_scheduler_type is None else lr_scheduler_type,
            "lr_scheduler_kwargs": get_lr_scheduler_kwargs(lr_scheduler_type),
            # "warmup_steps": int(0.08 * 10_000), # NOTE: ChemFormer: 8000
            # "warmup_ratio": warmup_ratio,
            "adam_beta1": 0.9, # NOTE: ChemFormer: 0.9
            "adam_beta2": 0.999, # NOTE: ChemFormer: 0.999
            "adam_epsilon": 1e-8, # Default: 1e-8
            # Batch size, device, and performance optimizations configs
            "batch_eval_metrics": False, # Default: False
            "group_by_length": True,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "auto_find_batch_size": True,
            "fp16": True if torch.cuda.is_available() else False,
            "fp16_full_eval" : True,  # Enable full BF16 evaluation for efficiency
            "half_precision_backend" : "auto",  # Let Hugging Face decide the best backend. Default: "auto"
            "use_cpu": False, # Default: False
            "dataloader_num_workers": 8, # Default: 0 (main process only)
            "dataloader_prefetch_factor": None, # Default: None
            # Evaluation and checkpointing configs
            "max_steps": max_steps,
            "num_train_epochs": num_train_epochs,
            "save_steps": 20_000, # NOTE: 200
            "save_strategy": "steps",
            "eval_steps": 20_000, # NOTE: 500
            "eval_delay": max(int(max(max_steps, num_train_epochs) * 0.7), 0), # Default: None
            "eval_strategy": "steps", # NOTE: "evaluation_strategy" is deprecated.
            "save_total_limit": 2, # This will save both the best and the last trainer checkpoint
            "load_best_model_at_end": True,
            "metric_for_best_model": "all_ligands_equal",
            "include_inputs_for_metrics": True,
            "eval_on_start": False, # Default: False
            # Logging configs
            "log_level": "debug",
            "logging_steps": 5000,
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
        if 'num_cycles' in training_args["lr_scheduler_kwargs"] and num_cycles is not None:
            training_args["lr_scheduler_kwargs"]["num_cycles"] = num_cycles
        if warmup_ratio is not None:
            training_args["warmup_ratio"] = warmup_ratio
        if warmup_steps is not None:
            training_args["warmup_steps"] = warmup_steps
            
        # Add Generation configs
        if causal_language_modeling:
            training_args["metric_for_best_model"] = "eval_loss"
        else:
            generation_config = GenerationConfig(
                max_length=512,
                max_new_tokens=512,
                do_sample=True,
                num_beams=5,
                temperature=1.0,
            )
            training_args["generation_config"] = generation_config
            training_args["predict_with_generate"] = True
            training_args["generation_config"] = generation_config
            training_args["generation_max_length"] = 512

    print("Training arguments:")
    for k, v in training_args.items():
        if 'token' in k:
            continue
        print(f"  - {k}: {v}")

    # Modify the training arguments with Optuna hyperparameter search
    if num_optuna_trials > 0:
        # Setup the compute_metrics function for the hyperparameter search
        hp_compute_metrics = partial(
            decode_and_get_metrics,
            tokenizer=tokenizer,
            compute_rdkit_metrics=False,
            compute_graph_metrics=False,
            num_proc=num_proc_map,
            causal_language_modeling=causal_language_modeling,
        )

        # Run the HP search (and update the training_args accordingly)
        best_run, hp_training_args = get_best_hyperparameters(
            model_init=model_lambda,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=hp_compute_metrics,
            dataset_tokenized=dataset_tokenized,
            training_args=copy.deepcopy(training_args),
            lr_scheduler_type=lr_scheduler_type,
            num_optuna_trials=num_optuna_trials,
            causal_language_modeling=causal_language_modeling,
            all_fragments_as_labels=all_fragments_as_labels,
            linkers_only_as_labels=linkers_only_as_labels,
        )
        best_objective = best_run.objective
        best_trial_number = best_run.run_id
        best_hparams = best_run.hyperparameters

        # Save to output directory the best hyperparameters
        with open(f"{output_dir}/best_hyperparameters.md", "w") as f:
            f.write(f"Number of Optuna trials: {num_optuna_trials}\n\n")
            f.write(f"Best trial objective: {best_objective:.4f} (best trial number: {best_trial_number})\n\n")

            f.write("Best hyperparameters:\n")
            for hparam, value in best_hparams.items():
                f.write(f"- {hparam}: {value}\n")
            f.write("\n")

            f.write("Training arguments:\n")
            for hparam, value in hp_training_args.items():
                if "token" in hparam:
                    continue
                elif isinstance(value, str):
                    if 'hf_' in value:
                        continue
                f.write(f"- {hparam}: {value}\n")
        
        # Open the file and remove any line that might contain the token
        with open(f"{output_dir}/best_hyperparameters.md", "r") as f:
            lines = f.readlines()
            with open(f"{output_dir}/best_hyperparameters.md", "w") as f:
                for line in lines:
                    if "hf_" in line:
                        continue
                    f.write(line)
        print(f"Best hyperparameters saved to '{output_dir}/best_hyperparameters.md'.")

        upload_single_file(
            path_or_fileobj=f"{output_dir}/best_hyperparameters.md",
            path_in_repo="best_hyperparameters.md",
            repo_id=hub_model_id,
            token=hub_token,
        )
        
        # Save the best_hparams to a JSON file
        with open(f"{output_dir}/best_hyperparameters.json", "w") as f:
            json.dump(best_hparams, f, indent=4)
        print(f"Best hyperparameters saved to '{output_dir}/best_hyperparameters.json'.")
        
        upload_single_file(
            path_or_fileobj=f"{output_dir}/best_hyperparameters.json",
            path_in_repo="best_hyperparameters.json",
            repo_id=hub_model_id,
            token=hub_token,
        )
        
        # Update the training arguments with the best hyperparameters
        hp_specific_args = [
            "num_train_epochs",
            "max_steps",
            "eval_steps",
            "eval_delay",
            "logging_steps",
            "save_steps",
            "generation_config",
        ]
        for k, v in hp_training_args.items():
            # Skip the specific arguments set/modifed by the HP search
            if k in hp_specific_args:
                continue
            training_args[k] = v

        # Update the num_cycles according to the original max_steps
        lr_scheduler_kwargs = hp_training_args["lr_scheduler_kwargs"]
        
        if "num_cycles" in lr_scheduler_kwargs:
            hp_num_cycles = lr_scheduler_kwargs["num_cycles"]
            hp_max_steps = hp_training_args["max_steps"]
            
            # Adjust/scale the max_cycles according to the number of steps
            if hp_max_steps > 0:
                hp_cycle_ratio = hp_num_cycles / hp_max_steps
                num_cycles = int(hp_cycle_ratio * max_steps)
                training_args["lr_scheduler_kwargs"]["num_cycles"] = num_cycles
                print(f"Adjusted number of cycles: {num_cycles}")

        # Adjust the warmup steps according to the original max_steps
        if "warmup_ratio" in hp_training_args:
            hp_warmup_ratio = hp_training_args["warmup_ratio"]
            hp_max_steps = hp_training_args["max_steps"]
            warmup_steps = int(hp_warmup_ratio * hp_max_steps)
            warmup_ratio = warmup_steps / max_steps
            training_args["warmup_steps"] = warmup_steps
            training_args["warmup_ratio"] = warmup_ratio

        print("Training arguments updated with the best hyperparameters:")
        for k, v in training_args.items():
            if 'token' in k:
                continue
            print(f"  - {k}: {v}")
        print("-" * 80)
        print("Starting training with the best hyperparameters.")
        print("-" * 80)

    # rouge = evaluate.load("rouge") # , cache_dir="/mimer/NOBACKUP/groups/naiss2023-6-290/stefano/.cache/huggingface/evaluate/")
    # fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
    #     radius=11,
    #     fpSize=1024,
    # )
    rouge = None
    fpgen = None
    compute_metrics = partial(
        decode_and_get_metrics,
        tokenizer=tokenizer,
        rouge=rouge,
        fpgen=fpgen,
        compute_rdkit_metrics=False,
        compute_graph_metrics=True,
        num_proc=max(1, num_proc_map - 2), # NOTE: Use 2 less process for the metrics, since there will be a timeout logic
        causal_language_modeling=causal_language_modeling,
    )

    if training_args_bin is not None:
        print(f"Loading training arguments from: {training_args_bin}.")
        # Load training arguments from a binary file and update model-specific arguments
        args = torch.load(training_args_bin)
        args.output_dir = output_dir
        args.overwrite_output_dir = True if delete_local_repo_if_exists else False
        args.push_to_hub_model_id = model_id
        args.push_to_hub_organization = organization
        args.hub_model_id = hub_model_id
        args.hub_token = hub_token
        # Print all the training arguments
        print("Training arguments loaded:")
        for k, v in args.__dict__.items():
            if 'token' in k:
                continue
            print(f"  - {k}: {v}")
    else:
        if causal_language_modeling:
            args = TrainingArguments(**training_args)
        else:
            args = Seq2SeqTrainingArguments(**training_args)

    if causal_language_modeling:
        TrainerClass = Trainer
    else:
        TrainerClass = Seq2SeqTrainer

    # Setup the Trainer and start training (no Optuna hyperparameter search)
    trainer = TrainerClass(
        model_init=model_lambda,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
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
    
    if causal_language_modeling:
        tasks = ["Text Generation"]
    else:
        tasks = ["Text2Text Generation", "question-answering"]

    if hub_model_id is not None:
        print("Pushing model to Hugging Face Hub.")
        print("-" * 80)
        tokenizer.save_pretrained(output_dir)
        trainer.push_to_hub(
            commit_message="Initial version",
            model_name=hub_model_id,
            license="mit",
            finetuned_from=f"{pretrained_encoder}",
            tasks=tasks,
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
