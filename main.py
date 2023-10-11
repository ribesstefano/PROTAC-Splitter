from train_model import train_model
from jsonargparse import CLI
from transformers import AutoTokenizer
from typing import Optional
import os

def main(
    batch_size: int = 128,
    batch_size_tokenizer: int = 512,
    gradient_accumulation_steps: int = 4,
    output_dir: str = "./models/",
    data_dir: str = "./data/final/",
    ds_name: Optional[str] = None,
    max_steps: int = -1,
    num_train_epochs: int = 40,
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_encoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_decoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    encoder_max_length: int = 256,
    decoder_max_length: int = 256,
    hub_token: Optional[str] = None,
    organization: Optional[str] = None,
):
    """Trains a model on a given dataset.

    Args:
        batch_size (int, optional): Batch size. Defaults to 128.
        batch_size_tokenizer (int, optional): Batch size for the tokenizer. Defaults to 512.
        gradient_accumulation_steps (int, optional): Gradient accumulation steps. Defaults to 4.
        output_dir (str, optional): Output directory. Defaults to "./models/".
        data_dir (str, optional): Data directory. If `ds_name` is not supplied, then a separate model will be trained on each subdirectory in `data_dir`. Each subdirectory shall include a HF-formatted dataset. Defaults to "./data/final/".
        ds_name (Optional[str], optional): Name of the dataset to train on. If `None`, then a separate model will be trained on each subdirectory in `data_dir`. Defaults to None.
        max_steps (int, optional): Maximum number of steps to train for. Defaults to -1.
        num_train_epochs (int, optional): Number of epochs to train for. Defaults to 40.
        tokenizer (AutoTokenizer | str, optional): Tokenizer to use. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_encoder (str, optional): Pretrained encoder to use. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_decoder (str, optional): Pretrained decoder to use. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        encoder_max_length (int, optional): Maximum length of the encoder input. Defaults to 256.
        decoder_max_length (int, optional): Maximum length of the decoder input. Defaults to 256.
        hub_token (Optional[str], optional): Hugging Face API token. Defaults to None.
        organization (Optional[str], optional): Organization to push the model to. Defaults to None.
    """
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    if ds_name is not None:
        train_model(
            ds_name=ds_name,
            data_dir=data_dir,
            batch_size_tokenizer= batch_size,
            batch_size=batch_size_tokenizer,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            output_dir=output_dir,
            tokenizer=tokenizer,
            pretrained_encoder="seyonec/ChemBERTa-zinc-base-v1",
            pretrained_decoder="seyonec/ChemBERTa-zinc-base-v1",
            encoder_max_length=encoder_max_length,
            decoder_max_length=decoder_max_length,
            hub_token=hub_token,
            organization=organization,
        )
    else:
        for f in os.listdir(os.fsencode(data_dir)):
            ds_name = os.fsdecode(f)
            train_model(
                ds_name=ds_name,
                data_dir=data_dir,
                batch_size_tokenizer= batch_size,
                batch_size=batch_size_tokenizer,
                max_steps=max_steps,
                num_train_epochs=num_train_epochs,
                gradient_accumulation_steps=gradient_accumulation_steps,
                output_dir=output_dir,
                tokenizer=tokenizer,
                pretrained_encoder="seyonec/ChemBERTa-zinc-base-v1",
                pretrained_decoder="seyonec/ChemBERTa-zinc-base-v1",
                encoder_max_length=encoder_max_length,
                decoder_max_length=decoder_max_length,
                hub_token=hub_token,
                organization=organization,
            )


if __name__ == '__main__':
    CLI(main)