from typing import Optional
from transformers import AutoTokenizer, EncoderDecoderModel
import torch

def get_model(
        pretrained_encoder: str = "seyonec/ChemBERTa-zinc-base-v1",
        pretrained_decoder: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_length: Optional[int] = 512,
        tie_encoder_decoder: bool = False,
) -> EncoderDecoderModel:
    """ Get the EncoderDecoderModel model for the PROTAC splitter.

    Args:
        pretrained_encoder (str): The pretrained model to use for the encoder. Default: "seyonec/ChemBERTa-zinc-base-v1"
        pretrained_decoder (str): The pretrained model to use for the decoder. Default: "seyonec/ChemBERTa-zinc-base-v1"
        max_length (int): The maximum length of the input sequence. Default: 512
        tie_encoder_decoder (bool): Whether to tie the encoder and decoder weights. Default: False

    Returns:
        EncoderDecoderModel: The EncoderDecoderModel model for the PROTAC splitter
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

    def setup_gen(config):
        config.do_sample = True
        config.num_beams = 5
        config.top_k = 20
        config.max_length = 512
        # config.max_new_tokens = 512
        return config
    
    bert2bert.config = setup_gen(bert2bert.config)
    bert2bert.encoder.config = setup_gen(bert2bert.encoder.config)
    bert2bert.decoder.config = setup_gen(bert2bert.decoder.config)
    bert2bert.decoder.config.is_decoder = True
    bert2bert.generation_config = setup_gen(bert2bert.generation_config)
    
    # bert2bert.config.do_sample = True
    # bert2bert.config.num_beams = 5
    # bert2bert.config.top_k = 20
    # bert2bert.config.max_length=512
    # bert2bert.config.max_new_tokens=512

    # bert2bert.generation_config.max_new_tokens = 512
    # bert2bert.generation_config.min_new_tokens = 512
    
    # bert2bert.config.max_new_tokens = 514
    # bert2bert.config.early_stopping = True
    # bert2bert.config.length_penalty = 2.0
    # # bert2bert.config.no_repeat_ngram_size = 3 # Default: 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert2bert.to(device)

    return bert2bert