from transformers import AutoTokenizer
from transformers import EncoderDecoderModel

def get_model(
    pretrained_encoder: str = "seyonec/ChemBERTa-zinc-base-v1",
    pretrained_decoder: str = "seyonec/ChemBERTa-zinc-base-v1",
):
    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
        pretrained_encoder,
        pretrained_decoder,
        tie_encoder_decoder=True,
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
    bert2bert.encoder.config.max_length = 514
    bert2bert.decoder.config.max_length = 514
    # bert2bert.config.min_length = 20
    # NOTE: Never sample, i.e., always return the token w/ highest probability
    bert2bert.config.do_sample = False
    
    # bert2bert.config.max_new_tokens = 514
    # bert2bert.config.early_stopping = True
    # bert2bert.config.length_penalty = 2.0
    # bert2bert.config.num_beams = 4
    # # bert2bert.config.no_repeat_ngram_size = 3 # Default: 0
    
    return bert2bert