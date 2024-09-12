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

from .data_utils import load_tokenized_dataset
from .evaluation import compute_metrics_with_chem
from .hf_utils import (
    create_hf_repository,
    delete_hf_repository,
    repo_exists,
)
from .model_utils import get_model


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
    training_args: Optional[Seq2SeqTrainingArguments | Dict[str, Any]] = None,
    resume_from_checkpoint: Optional[str] = None,
    optuna_n_trials: int = 0,
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
        optuna_n_trials (int, optional): The number of Optuna trials. Defaults to 0, i.e., no Optuna hyperparameter search.
    """
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
    # try:
    #     bert2bert = EncoderDecoderModel.from_pretrained(hub_model_id)
    #     print(f"Skipping pretrained model {hub_model_id}.")
    # except:
    #     print('-' * 80)
    #     print(f"Training model {hub_model_id} on dataset: {ds_name}.")
    #     print('-' * 80)
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
        generation_config = GenerationConfig(
            max_length=512,
            do_sample=True,
            num_beams=5,
            temperature=1.0,
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            # Optimizer-related configs
            learning_rate=learning_rate,
            optim="adamw_torch",
            lr_scheduler_type="cosine", # Default: "linear"
            warmup_ratio=0.05,
            # Generation configs
            predict_with_generate=True,
            generation_num_beams=5, # Greedy strategy
            generation_config=generation_config,
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
            eval_steps=5, # NOTE: 100
            save_steps=200,
            # eval_steps=7500,
            # warmup_steps=2000,
            save_strategy="steps",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            include_inputs_for_metrics=True,
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
            report_to=["tensorboard"],
            # Other configs
            seed=42,
            data_seed=42,
        )
    elif isinstance(training_args, dict):
        training_args = Seq2SeqTrainingArguments(**training_args)
    rouge = evaluate.load("rouge")
    fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=11,
        fpSize=1024,
    )
    metric = partial(
        compute_metrics_with_chem,
        rouge=rouge,
        tokenizer=tokenizer,
        fpgen=fpgen,
    )
    bert2bert = lambda: get_model(
        pretrained_encoder=pretrained_encoder,
        pretrained_decoder=pretrained_decoder,
        max_length=encoder_max_length,
        tie_encoder_decoder=tie_encoder_decoder,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=bert2bert)
    trainer = Seq2SeqTrainer(
        model_init=bert2bert,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=metric,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
    )
    if optuna_n_trials > 0:
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
                "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
            }

        def compute_objective(metrics: Dict[str, float]):
            return metrics["eval_loss"], metrics["eval_reassembly"]

        best_trials = trainer.hyperparameter_search(
            direction=["minimize", "maximize"],
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=optuna_n_trials,
            compute_objective=compute_objective,
        )
        print("-" * 80)
        print(f"Best trials:\n{best_trials}")
        print("-" * 80)
    else:
        trainer.train(
            resume_from_checkpoint=resume_from_checkpoint, # "last-checkpoint",
        )
    if hub_model_id is not None:
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
        # tokenizer.push_to_hub(
        #     repo_id=hub_model_id,
        #     commit_message="Upload tokenizer",
        #     private=True,
        #     token=hub_token,
        #     tags=["PROTAC", "cheminformatics"],
        # )
