"""Evaluate language model perplexity on PROTAC splitting predictions.

Loads an encoder-decoder Transformer from HuggingFace, runs generation with
multiple strategies, and computes perplexity for each predicted sequence.

Usage:
    python scripts/evaluate_perplexity.py --help
    python scripts/evaluate_perplexity.py --model-name ailab-bio/PROTAC-Splitter-EncoderDecoder-lr_reduce-rand-smiles
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from scripts.common import ensure_output_dir, get_hub_token, load_dataset_or_csv


@dataclasses.dataclass
class Args:
    """Evaluate encoder-decoder Transformer perplexity on PROTAC splitting."""

    model_name: str = "ailab-bio/PROTAC-Splitter-EncoderDecoder-lr_reduce-rand-smiles"
    """HuggingFace model name or local path."""

    hub_token: Optional[str] = None
    batch_size: int = 32

    dataset_id: str = "ailab-bio/PROTAC-Splitter-Dataset"
    dataset_config: str = "clustered"
    dataset_split: str = "held_out"
    input_csv: Optional[str] = None
    """Local CSV file (overrides HuggingFace Hub if provided)."""

    output_csv: str = "logs/perplexity_results.csv"
    cache_dir: Optional[str] = None

    generation_strategies: str = "beam_search_decoding,greedy"
    """Comma-separated list of generation strategies to evaluate."""


def main(args: Args) -> None:
    import torch
    from datasets import Dataset
    from tqdm import tqdm
    from transformers import AutoTokenizer, EncoderDecoderModel, GenerationConfig
    from torchmetrics.text import Perplexity

    token = get_hub_token(args.hub_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_ds = load_dataset_or_csv(
        args.input_csv,
        hub_dataset_id=args.dataset_id,
        hub_config=args.dataset_config,
        hub_split=args.dataset_split,
        hub_token=token,
        cache_dir=args.cache_dir,
    )
    print(f"Loaded {len(test_ds)} samples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)
    model = EncoderDecoderModel.from_pretrained(args.model_name, token=token).to(device).eval()
    perplexity_metric = Perplexity(ignore_index=tokenizer.pad_token_id).to(device)

    strategies_map = {
        "greedy": GenerationConfig(max_new_tokens=512, num_beams=1, do_sample=False),
        "beam_search_decoding": GenerationConfig(max_new_tokens=512, num_beams=5, do_sample=False, num_return_sequences=5),
        "multinomial_sampling": GenerationConfig(max_new_tokens=512, num_beams=1, do_sample=True),
        "contrastive_search": GenerationConfig(max_new_tokens=512, penalty_alpha=0.1, top_k=10),
    }
    selected_strategies = [s.strip() for s in args.generation_strategies.split(",")]

    all_rows = []

    for strategy_name in selected_strategies:
        if strategy_name not in strategies_map:
            print(f"Unknown strategy '{strategy_name}', skipping.")
            continue
        gen_config = strategies_map[strategy_name]
        print(f"\n--- Strategy: {strategy_name} ---")

        for i in tqdm(range(0, len(test_ds), args.batch_size), desc=f"Perplexity ({strategy_name})"):
            idxs = list(range(i, min(i + args.batch_size, len(test_ds))))
            batch_texts = [test_ds[j]["text"] for j in idxs]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model.generate(**enc, generation_config=gen_config,
                                         output_scores=True, return_dict_in_generate=True)

            probs = torch.exp(outputs.sequences_scores).tolist() if hasattr(outputs, "sequences_scores") else [None] * len(idxs)
            dec_ids = outputs.sequences
            labels = dec_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            dec_mask = (dec_ids != tokenizer.pad_token_id).long()

            with torch.no_grad():
                logits = model(**enc, decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask, labels=labels).logits

            for j, (gl, gt, prob) in enumerate(zip(logits, dec_ids, probs)):
                seq_str = tokenizer.decode(gt, skip_special_tokens=True)
                ppl = perplexity_metric(preds=gl.unsqueeze(0)[:, :-1], target=gt.unsqueeze(0)[:, 1:]).item()
                all_rows.append({
                    "text": batch_texts[j % len(batch_texts)],
                    "generated": seq_str,
                    "probability": prob,
                    "perplexity": ppl,
                    "strategy": strategy_name,
                    "model_name": args.model_name,
                })

    out = Path(args.output_csv)
    ensure_output_dir(str(out.parent))
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nPerplexity results saved: {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
