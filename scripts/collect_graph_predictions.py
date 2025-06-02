import argparse
import pandas as pd
from protac_splitter.graphs.splitting_algorithms import split_protac_with_graphs
from protac_splitter.graphs.e3_clustering import get_representative_e3s
from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier # To make joblib find the model class

def main():
    parser = argparse.ArgumentParser(description="Split PROTACs using graph-based algorithms and output predictions as CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file with PROTAC SMILES.")
    parser.add_argument("--smiles_column", type=str, default="text", help="Column name for PROTAC SMILES in input CSV.")
    parser.add_argument("--labels_column", type=str, default="labels", help="Column name for label SMILES in input CSV.")
    parser.add_argument("--classifier_model", type=str, default=None, help="Path to the edge classifier model.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file for predictions.")
    parser.add_argument("--use_classifier", action="store_true", default=True, help="Use edge classifier for splitting (default: True).")
    parser.add_argument("--no_use_classifier", action="store_false", dest="use_classifier", help="Do not use edge classifier.")
    parser.add_argument("--representative_e3s_csv", type=str, default=None, help="Optional CSV file with representative E3s.")
    parser.add_argument("--e3_column", type=str, default="E3 Binder SMILES with direction", help="Column name for E3s in representative E3s CSV.")
    parser.add_argument("--run_e3_clustering", action="store_true", help="Run get_representative_e3s on the input CSV to generate representative E3s.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for parallel jobs.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    # Expecting columns: 'text' (protac_smiles), 'labels' (label_smiles)
    smiles_list = df[args.smiles_column].tolist()
    labels_list = df[args.labels_column].tolist()

    representative_e3s = None
    if args.run_e3_clustering:
        # Run clustering on the input CSV to get representative E3s
        representative_e3s, _, _, _ = get_representative_e3s(df, e3_column=args.e3_column)
    elif args.representative_e3s_csv:
        # Read E3s from a separate CSV file
        e3_df = pd.read_csv(args.representative_e3s_csv)
        representative_e3s = e3_df[args.e3_column].dropna().unique().tolist()

    if args.classifier_model is None:
        # Raise a warning if the classifier model is not provided
        if args.use_classifier:
            raise ValueError("Edge classifier model is required when --use_classifier (default ON) is set.")
        else:
            print("No edge classifier model provided, proceeding without edge classification.")

    print(f"Starting splitting {len(smiles_list)} PROTACs with graph-based algorithms...")

    results = split_protac_with_graphs(
        smiles_list,
        use_classifier=args.use_classifier,
        classifier=args.classifier_model,
        representative_e3s=representative_e3s,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
    )

    # Determine model name based on whether classifier is used
    model_name = "GraphEdgeClassifier" if args.use_classifier else "GraphHeuristic"

    # Prepare output DataFrame in the requested format
    out_df = pd.DataFrame([
        {
            "protac_smiles": smi,
            "label_smiles": label,
            "default_pred_n0": f"{ret['e3']}.{ret['linker']}.{ret['poi']}",
            "model_name": model_name,
        }
        for smi, label, ret in zip(smiles_list, labels_list, results)
    ])
    out_df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
