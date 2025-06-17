import os
import argparse
import logging

import pandas as pd
from datasets import Dataset, load_dataset

from protac_splitter.graphs.splitting_algorithms import split_protac_graph_based
from protac_splitter.graphs.e3_clustering import get_representative_e3s_fp
from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier
from protac_splitter.llms.model_utils import get_pipeline, run_pipeline
from protac_splitter.protac_splitter import fix_prediction
from protac_splitter.evaluation import check_reassembly

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect combined predictions for PROTACs.")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV file with PROTACs.")
    parser.add_argument("--output_csv", type=str, default="combined_models-preds.csv", help="Output CSV file for predictions.")
    parser.add_argument("--smiles_column", type=str, default="text", help="Column name for PROTAC SMILES.")
    parser.add_argument("--labels_column", type=str, default="labels", help="Column name for label SMILES.")
    parser.add_argument("--use_classifier", action='store_true', help="Use edge classifier model.")
    parser.add_argument("--classifier_model", type=str, help="Path to edge classifier model.")
    parser.add_argument("--use_capacity_weight", action='store_true', help="Use capacity weight in graph splitting.")
    parser.add_argument("--betweenness_threshold", type=float, default=0.5, help="Betweenness threshold for graph splitting.")
    parser.add_argument("--model_name", type=str, default="ailab-bio/PROTAC-Splitter-EncoderDecoder-lr_reduce-rand-smiles", help="Model name for seq2seq pipeline.")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face Hub token for accessing private models.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache models and tokenizers.")
    parser.add_argument("--dataset_dir", type=str, default="ailab-bio/PROTAC-Splitter-Dataset", help="Directory to save the dataset.")
    parser.add_argument("--dataset_config", type=str, default="clustered", help="Dataset configuration for the seq2seq pipeline.")
    parser.add_argument("--dataset_test_split", type=str, default="held_out", help="Test split of the dataset for seq2seq pipeline.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--is_causal_language_model", action='store_true', help="Set to True if using a causal language model (e.g., GPT).")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of parallel jobs.")

    args = parser.parse_args()

    # Set log level to ERROR
    logging.basicConfig(level=logging.ERROR)

    # Check if hub_token is provided
    if args.hub_token is None:
        args.hub_token = os.getenv('HF_TOKEN', None)
        if args.hub_token is None:
            raise ValueError('Hugging Face API token not provided. Please provide a token using the --hub_token argument or set the HF_TOKEN environment variable')

    print('Loading dataset...')
    # Load the input CSV file
    if args.input_csv is not None:
        print(f"Loading input CSV file: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        test_ds = Dataset.from_pandas(df)
    else:
        if os.path.exists(args.dataset_dir):
            print(f"Loading dataset from directory: {args.dataset_dir}")
            test_ds = load_dataset(
                args.dataset_dir,
                data_dir=args.dataset_config,
            )[args.dataset_test_split]
        else:
            print(f"Loading dataset from Hugging Face Hub: {args.dataset_dir}")
            test_ds = load_dataset(
                args.dataset_dir,
                args.dataset_config,
                token=args.hub_token,
                cache_dir=args.cache_dir,
            )[args.dataset_test_split]

    pipe = get_pipeline(
        model_name=args.model_name,
        token=args.hub_token,
        is_causal_language_model=args.is_causal_language_model,
    )
    print(f"Pipeline loaded with model: {args.model_name}")

    # Getting a list of predictions
    preds = run_pipeline(pipe, test_ds, args.batch_size, args.is_causal_language_model, args.smiles_column)
    print(f"Generated predictions for {len(preds)} samples.")

    # Add the list of predictions to the dataset
    test_ds = test_ds.add_column("predictions", preds)

    representative_e3s_fp = get_representative_e3s_fp(verbose=1)

    if args.use_classifier:
        # Load the edge classifier model
        classifier = GraphEdgeClassifier.load(args.classifier_model)
        model_name = "TransformerAndGraphEdgeClassifier"
        print(f"Using edge classifier model: {args.classifier_model}")
    else:
        classifier = None
        model_name = "TransformerAndGraphHeuristic"
        print("Using heuristic-based algorithm without edge classifier model.")

    def mapping_func(example):
        protac_smiles = example[args.smiles_column]
        pred_smiles = example["predictions"]["pred_n0"] # Take top-1 prediction

        fixed_smiles = fix_prediction(protac_smiles, pred_smiles)
        if not check_reassembly(protac_smiles, fixed_smiles):
            ret = split_protac_graph_based(
                protac_smiles=protac_smiles,
                use_classifier=args.use_classifier,
                classifier=classifier,
                representative_e3s_fp=representative_e3s_fp,
                morgan_fp_generator=None,
                use_capacity_weight=args.use_capacity_weight,
                betweenness_threshold=args.betweenness_threshold,
            )
            fixed_smiles = f"{ret['e3']}.{ret['linker']}.{ret['poi']}"

        return {
            "protac_smiles": protac_smiles,
            "label_smiles": example[args.labels_column],
            "default_pred_n0": fixed_smiles,
            "model_name": model_name,
        }

    # Apply the mapping function to the dataset
    print("Calling XGBoost-based graph splitting algorithm for failed predictions...")
    out_df = test_ds.map(
        mapping_func,
        remove_columns=[args.smiles_column, args.labels_column, "predictions"],
        num_proc=args.num_proc,
    ).to_pandas()
    print(f"Finished splitting PROTACs. Output DataFrame saved to: {args.output_csv}")

    # Save the output DataFrame to CSV
    if args.output_csv is None:
        print(f"No output CSV specified. Using default: {args.output_csv}")
    out_df.to_csv(args.output_csv, index=False)