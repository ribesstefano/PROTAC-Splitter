from model_utils import get_model
from data_utils import load_tokenized_dataset
import os
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EncoderDecoderModel,
    AutoTokenizer,
)
import evaluate
from typing import Optional
from functools import partial

import huggingface_hub as hf

def create_hf_repository(**kwargs):
  """Creates a new Hugging Face repository."""

  api = hf.HfApi()
  return api.create_repo(**kwargs)


def compute_metrics(
    pred,
    rouge = evaluate.load("rouge"),
    tokenizer: [AutoTokenizer, str] = "seyonec/ChemBERTa-zinc-base-v1",
):
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    return {k: round(v, 4) for k, v in rouge_output.items()}


def train_model(
    ds_name: str,
    max_steps: int = -1,
    num_train_epochs: int = 40,
    batch_size: int = 128,
    batch_size_tokenizer: int = 512,
    gradient_accumulation_steps: int = 4,
    hub_token: Optional[str] = None,
    organization: Optional[str] = None,
    output_dir: str = "./models/",
    data_dir: str = './data/final/',
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_encoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_decoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    encoder_max_length: int = 256,
    decoder_max_length: int = 256,
):
    """Trains a model on a given dataset.
    
    Args:
        ds_name (str): Name of the dataset to train on.
        max_steps (int, optional): Maximum number of steps to train for. Defaults to -1.
        num_train_epochs (int, optional): Number of epochs to train for. Defaults to 40.
        batch_size (int, optional): Batch size. Defaults to 128.
        batch_size_tokenizer (int, optional): Batch size for the tokenizer. Defaults to 512.
        gradient_accumulation_steps (int, optional): Number of gradient accumulation steps. Defaults to 4.
        output_dir (str, optional): Output directory. Defaults to "./models/".
        data_dir (str, optional): Directory containing the dataset. Defaults to "./data/final/".
        tokenizer (AutoTokenizer | str, optional): Tokenizer to use. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_encoder (str, optional): Pretrained encoder model to use. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_decoder (str, optional): Pretrained decoder model to use. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        encoder_max_length (int, optional): Maximum length of the encoder input. Defaults to 256.
        decoder_max_length (int, optional): Maximum length of the decoder input. Defaults to 256.
        organization (Optional[str], optional): Organization to push the model to. Defaults to None.
        hub_token (Optional[str], optional): Hugging Face API token. Defaults to None.
    """
    model_id = "PROTAC-Splitter" + "_".join((ds_name.replace("protac_splitter", "PROTAC-Splitter").split("_")[1:])).replace("%", "perc")
    # model_id = ds_name.replace("protac_splitter", "PROTAC-Splitter").replace("%", "perc")
    output_dir += f"/{model_id}"
    if organization is not None:
        hub_model_id = f"{organization}/{model_id}"
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
    bert2bert = get_model(pretrained_encoder, pretrained_decoder)
    dataset_tokenized = load_tokenized_dataset(
        os.path.join(data_dir, ds_name),
        tokenizer,
        batch_size_tokenizer,
        encoder_max_length,
        decoder_max_length,
        token=hub_token,
    )
    per_device_batch_size = batch_size // gradient_accumulation_steps
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # Optimizer-related configs
        learning_rate=5e-5,
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
        save_steps=500,
        # eval_steps=7500,
        # warmup_steps=2000,
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        # Logging configs
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
    trainer = Seq2SeqTrainer(
        model=bert2bert,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=partial(compute_metrics, rouge=rouge, tokenizer=tokenizer),
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
            finetuned_from=f"Encoder: {pretrained_encoder}, Decoder: {pretrained_decoder}",
            tasks=["Text2Text Generation"],
            tags=["PROTAC", "cheminformatics"],
        )