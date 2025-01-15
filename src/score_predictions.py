from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from protac_splitter.evaluation import score_prediction

def main():

    # Get all files with "*preds.csv" in the logs directory
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs directory not found")
    
    # Get all predictions CSV files
    predictions_files = list(logs_dir.glob("*preds.csv"))
    if not predictions_files:
        raise FileNotFoundError("No predictions CSV files found in logs directory")
    
    print(f"Found {len(predictions_files)} predictions CSV files in logs directory: {predictions_files}")

    for predictions_path in predictions_files:
        print(f"Scoring predictions in {predictions_path}")
        print(f"Model name: {predictions_path.stem}")
        scores_path = predictions_path.with_name(predictions_path.stem.replace("preds", "scores") + ".csv")
        print(f"Scores will be saved in {scores_path}")

        # Read predictions CSV
        df = pd.read_csv(predictions_path)

        # Remove rows in which the label_smiles has more than two dots
        df = df[~df['label_smiles'].str.contains("\.\[Cl-\]\.")]

        # Convert the pandas DataFrame to a Hugging Face Dataset
        ds = Dataset.from_pandas(df, preserve_index=False)
        
        def score_multiple_predictions(row):
            # Get the number of return sequences based on the number of columns that start with "default_pred_"
            num_return_sequences = sum([1 for c in row.keys() if c.startswith("default_pred_")])

            scores = {}
            scores['protac_smiles'] = row['protac_smiles']
            scores['label_smiles'] = row['label_smiles']
            for i in range(num_return_sequences):
                curr_scores = score_prediction(
                    protac_smiles=row['protac_smiles'],
                    label_smiles=row['label_smiles'], 
                    pred_smiles=row[f'default_pred_{i}'],
                    compute_graph_metrics=True,
                    graph_edit_kwargs={'timeout': 0.1}
                )
                curr_scores = {f"pred_n{i}_{k}": v for k, v in curr_scores.items()}
                scores.update(curr_scores)
            
            # Remove unused score metrics: tanimoto
            scores = {k: v for k, v in scores.items() if 'tanimoto' not in k}
            return scores

        # Use the map function to apply the scoring function to each row
        scores = ds.map(
            score_multiple_predictions,
            num_proc=16,
        )

        # Convert the scores to a pandas DataFrame and save it to file
        pd.DataFrame(scores).to_csv(scores_path, index=False)

    return

    # Read predictions CSV
    predictions_path = Path("logs/predictions.csv")
    if not predictions_path.exists():
        raise FileNotFoundError("predictions.csv not found in logs directory")
        
    df = pd.read_csv(predictions_path)

    # Remove rows in which the label_smiles has more than two dots
    df = df[~df['label_smiles'].str.contains("\.\[Cl-\]\.")]
    
    # preds_configs = [c for c in df.columns if c.startswith("preds_")]

    # Apply scoring function to each row
    scores = []
    for i, row in tqdm(df.iterrows(), desc="Scoring predictions", total=len(df)):
        # for config in preds_configs:
        config = "preds_default"
        score = score_prediction(
            protac_smiles=row['protac_smiles'],
            label_smiles=row['label_smiles'], 
            pred_smiles=row[config],
            compute_graph_metrics=True,
            graph_edit_kwargs={'timeout': 0.1}
        )
        score['protac_smiles'] = row['protac_smiles']
        score['label_smiles'] = row['label_smiles']
        score['pred_smiles'] = row[config]
        score['preds_config'] = config
        scores.append(score)

    # Save scored predictions
    pd.DataFrame(scores).to_csv("logs/scores.csv", index=False)
    
if __name__ == "__main__":
    main()