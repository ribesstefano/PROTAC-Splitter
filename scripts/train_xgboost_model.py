import os
import argparse

from datasets import load_dataset
import pandas as pd
from protac_splitter.graphs.edge_classifier import train_edge_classifier

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "..", "models")

def main():
    parser = argparse.ArgumentParser(description="Train an edge classifier for PROTAC splitting.")
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="./models/PROTAC-Splitter-XGBoost.joblib",
        help="Path to save the trained edge classifier model.",
    )
    parser.add_argument(
        "--graph_datasets_cache_dir",
        type=str,
        default="./data/graph_based/",
        help="Directory to cache the graph datasets.",
    )

    args = parser.parse_args()
    
    # Check if train.csv exists in the specified cache directory
    train_csv_path = f"{args.graph_datasets_cache_dir}/train.csv"
    if not os.path.exists(train_csv_path):
        # Load the dataset
        ds = load_dataset('ailab-bio/PROTAC-Splitter-Dataset', 'clustered')
        def get_substructs(row):
            text = row["text"]
            labels = row["labels"]
            return {
                "PROTAC SMILES": text,
                "POI Ligand SMILES with direction": labels.split(".")[2],
                "Linker SMILES with direction": labels.split(".")[1],
                "E3 Binder SMILES with direction": labels.split(".")[0],
            }

        ds = load_dataset("ailab-bio/PROTAC-Splitter-Dataset", "clustered")
        ds = ds.map(get_substructs, num_proc=8, remove_columns=["text", "labels"])
        train_df = ds["train"].to_pandas()
        val_df = ds["validation"].to_pandas()
        test_df = ds["test"].to_pandas()
    else:
        print(f"Loading training data from: {train_csv_path}")
        train_df = None
        val_df = None
        test_df = None

    train_edge_classifier(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        model_filename=args.output_model_path,
        cache_dir=args.graph_datasets_cache_dir,
    )
    print(f"Edge classifier model saved to: {args.output_model_path}")


if __name__ == "__main__":
    main()