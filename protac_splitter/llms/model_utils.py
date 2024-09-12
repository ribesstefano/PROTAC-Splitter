from typing import Optional
from transformers import AutoTokenizer, EncoderDecoderModel


def get_model(
    pretrained_encoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_decoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_length: Optional[int] = 512,
    tie_encoder_decoder: bool = False,
    do_sample: bool = True,
) -> EncoderDecoderModel:
    """Get an EncoderDecoder model for training.

    Args:
        pretrained_encoder (str, optional): The name of the pretrained encoder. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        pretrained_decoder (str, optional): The name of the pretrained decoder. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        max_length (Optional[int], optional): The maximum length. Defaults to 512. NOTE: Only used in generation, the maximum length of the input is determined by the positional embeddings.
        tie_encoder_decoder (bool, optional): Whether to tie the encoder and decoder weights. Defaults to False.
        do_sample (bool, optional): Whether to sample. Defaults to True, i.e., 5 beams and top-k sampling (k = 20).

    Returns:
        EncoderDecoderModel: The model.
    """
    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
        pretrained_encoder,
        pretrained_decoder,
        tie_encoder_decoder=tie_encoder_decoder,
    )
    print(f"Number of parameters: {bert2bert.num_parameters():,}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_encoder)

    # Tokenizer configs
    bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
    bert2bert.config.eos_token_id = tokenizer.sep_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id
    bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

    # Generation configs
    # NOTE: See full list of configurations can be found here: https://huggingface.co/docs/transformers/v4.33.3/en/main_classes/text_generation#transformers.GenerationConfig
    bert2bert.encoder.config.max_length = max_length
    bert2bert.decoder.config.max_length = max_length
    # bert2bert.config.min_length = 20

    if do_sample:
        bert2bert.config.do_sample = True
        bert2bert.config.num_beams = 5
        bert2bert.config.top_k = 20
    else:
        bert2bert.config.do_sample = False
        bert2bert.config.num_beams = 1
    
    # bert2bert.config.max_new_tokens = 514
    # bert2bert.config.early_stopping = True
    # bert2bert.config.length_penalty = 2.0
    # # bert2bert.config.no_repeat_ngram_size = 3 # Default: 0
    
    return bert2bert