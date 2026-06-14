"""Collect Transformer predictions for PROTAC splitting evaluation.

Usage:
    python scripts/collect_llm_predictions.py --help
    python scripts/collect_llm_predictions.py --model-name ailab-bio/PROTAC-Splitter-EncoderDecoder-lr_reduce-rand-smiles
"""
from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tyro
from rdkit import RDLogger, rdBase
from tqdm import tqdm

from scripts.common import ensure_output_dir, get_hub_token

RDLogger.DisableLog("rdApp.*")
blocker = rdBase.BlockLogs()

GENERATION_STRATEGY_PARAMS = {
    "greedy": {"num_beams": 1, "do_sample": False},
    "contrastive_search": {"penalty_alpha": 0.1, "top_k": 10},
    "multinomial_sampling": {"num_beams": 1, "do_sample": True},
    "beam_search_decoding": {"num_beams": 5, "do_sample": False, "num_return_sequences": 5},
    "beam_search_multinomial_sampling": {"num_beams": 5, "do_sample": True, "num_return_sequences": 5},
    "diverse_beam_search_decoding": {"num_beams": 5, "num_beam_groups": 5, "diversity_penalty": 1.0, "num_return_sequences": 5},
}


@dataclasses.dataclass
class Args:
    """Collect Transformer model predictions on the PROTAC-Splitter test set."""

    model_name: str = "ailab-bio/PROTAC-Splitter-standard_recombined-ChemBERTa-zinc-base-v1"
    """HuggingFace model name or path."""

    hub_token: Optional[str] = None
    """HuggingFace API token (defaults to HF_TOKEN in .env)."""

    batch_size: int = 64
    log_dir: str = "logs"
    num_proc: int = 8

    eval_gen_strategies: bool = False
    """Evaluate all generation strategies (beam search, etc.)."""

    report_model_name: Optional[str] = None
    """Short name for the model used in output filenames."""

    cache_dir: str = "~/.cache/huggingface"
    is_causal_language_model: bool = False
    get_predictions_probabilities: bool = False

    dataset_dir: str = "ailab-bio/PROTAC-Splitter-Dataset"
    dataset_config: str = "clustered"
    dataset_test_split: str = "held_out"


def _get_generation_config(strategy: str):
    from transformers import GenerationConfig
    return GenerationConfig(max_length=512, max_new_tokens=512, **GENERATION_STRATEGY_PARAMS[strategy])


def _get_pipeline(model_name, token, is_causal, generation_strategy=None):
    import torch
    from transformers import AutoTokenizer, GenerationConfig, pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_kwargs = dict(token=token)
    if is_causal:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **tokenizer_kwargs)
        kwargs = dict(model=model_name, tokenizer=tokenizer, token=token, device=device, num_return_sequences=1)
        if generation_strategy:
            kwargs["generation_config"] = _get_generation_config(generation_strategy)
        return pipeline("text-generation", **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        kwargs = dict(model=model_name, tokenizer=tokenizer, token=token, device=device)
        if generation_strategy:
            kwargs["generation_config"] = _get_generation_config(generation_strategy)
        return pipeline("text2text-generation", **kwargs)


def _run_pipeline(pipe, test_ds, batch_size, is_causal) -> List[Dict]:
    from transformers.pipelines.pt_utils import KeyDataset
    preds = []
    if is_causal:
        for pred in tqdm(pipe(KeyDataset(test_ds, "prompt"), batch_size=batch_size, max_length=512), total=len(test_ds) // batch_size):
            text = [p["generated_text"] for p in pred]
            text = [".".join(t.split(".")[1:]) for t in text]
            preds.append({f"pred_n{i}": t for i, t in enumerate(text)})
    else:
        for pred in tqdm(pipe(KeyDataset(test_ds, "text"), batch_size=batch_size, max_length=512), total=len(test_ds) // batch_size):
            preds.append({f"pred_n{i}": p["generated_text"] for i, p in enumerate(pred)})
    return preds


def main(args: Args) -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, EncoderDecoderModel
    from torchmetrics.text import Perplexity

    logging.basicConfig(level=logging.ERROR)
    token = get_hub_token(args.hub_token)

    print("Loading dataset...")
    ds_path = Path(args.dataset_dir)
    if ds_path.exists():
        test_ds = load_dataset(str(ds_path), data_dir=args.dataset_config)[args.dataset_test_split]
    else:
        test_ds = load_dataset(
            args.dataset_dir, args.dataset_config, token=token,
            cache_dir=args.cache_dir,
        )[args.dataset_test_split]

    report_name = args.report_model_name or next(
        n for n in args.model_name.split("/") if "PROTAC-Splitter" in n
    )
    print(f"Model: {args.model_name}  |  report name: {report_name}")

    preds: dict = defaultdict(list)

    if args.is_causal_language_model:
        test_ds = test_ds.map(
            lambda x: {"text": x["text"], "prompt": x["text"] + ".", "labels": x["labels"]},
            num_proc=args.num_proc,
        )

    pipe = _get_pipeline(args.model_name, token, args.is_causal_language_model)

    if args.get_predictions_probabilities:
        if args.is_causal_language_model:
            raise ValueError("Prediction probabilities not supported for causal LMs.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)
        perplexity = Perplexity(ignore_index=tokenizer.pad_token_id).to(device)
        model = EncoderDecoderModel.from_pretrained(args.model_name, token=token).to(device).eval()

        for i in tqdm(range(0, len(test_ds), args.batch_size), desc="Probabilities"):
            idxs = list(range(i, min(i + args.batch_size, len(test_ds))))
            batch = tokenizer(test_ds.select(idxs)["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model.generate(**batch, output_scores=True, return_dict_in_generate=True)
            probs = torch.exp(outputs.sequences_scores).tolist()
            dec_ids = outputs.sequences
            labels = dec_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            dec_mask = torch.ones_like(dec_ids)
            dec_mask[dec_ids == tokenizer.pad_token_id] = 0
            with torch.no_grad():
                logits = model(**batch, decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask, labels=labels).logits
            for gl, gt, gp in zip(logits, dec_ids, probs):
                s = tokenizer.decode(gt, skip_special_tokens=True)
                ppl = perplexity(preds=gl.unsqueeze(0)[:, :-1], target=gt.unsqueeze(0)[:, 1:]).item()
                preds["default"].append({"pred_n0": s, "prob_n0": gp, "perplexity_n0": ppl})
    else:
        preds["default"] = _run_pipeline(pipe, test_ds, args.batch_size, args.is_causal_language_model)

    if args.eval_gen_strategies:
        for strategy in GENERATION_STRATEGY_PARAMS:
            print(f"Strategy: {strategy}")
            p = _get_pipeline(args.model_name, token, args.is_causal_language_model, strategy)
            preds[strategy] = _run_pipeline(p, test_ds, args.batch_size, args.is_causal_language_model)

    rows = []
    for i, (text, labels) in enumerate(zip(test_ds["text"], test_ds["labels"])):
        row = {"protac_smiles": text, "label_smiles": labels}
        for strategy, predictions in preds.items():
            row.update({f"{strategy}_{k}": v for k, v in predictions[i].items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    df["model_name"] = args.model_name

    log_dir = ensure_output_dir(args.log_dir)
    out_path = log_dir / f"{report_name}-preds.csv"
    df.to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
