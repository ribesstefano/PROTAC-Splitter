import os
from pathlib import Path
import logging
import argparse

import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from protac_splitter.evaluation import score_prediction
from protac_splitter.protac_splitter import fix_prediction

def main(
    num_proc: int = 16,
):

    # Set logging level to error
    logging.basicConfig(level=logging.ERROR) # , force=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    # Get all files with "*preds.csv" in the logs directory
    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs directory not found")
    
    # Get all predictions CSV files
    predictions_files = list(logs_dir.glob("*preds.csv"))
    if not predictions_files:
        raise FileNotFoundError("No predictions CSV files found in logs directory")
    
    print(f"Found {len(predictions_files)} predictions CSV files in logs directory:")
    for f in predictions_files:
        print(f" - {f}")
    print()

    for predictions_path in predictions_files:

        # if 'Trial' not in str(predictions_path.stem):
        #     print(f"Skipping file: {predictions_path}")
        #     continue

        print(f"Scoring predictions in {predictions_path}")
        print(f"Model name: {predictions_path.stem}")
        scores_path = predictions_path.with_name(predictions_path.stem.replace("preds", "scores") + ".csv")
        print(f"Scores will be saved in {scores_path}")

        # if os.path.exists(scores_path):
        #     print(f"Scores already exist for {predictions_path.stem}, skipping...")
        #     continue

        # Read predictions CSV
        df = pd.read_csv(predictions_path)

        # Remove rows in which the label_smiles has more than two dots
        df = df[~df['label_smiles'].str.contains("\.\[Cl-\]\.")]

        # Convert the pandas DataFrame to a Hugging Face Dataset
        ds = Dataset.from_pandas(df, preserve_index=False)
        
        def score_multiple_predictions(row):
            # Get the number of return sequences based on the number of columns that start with "default_pred_n"
            num_return_sequences = sum([1 for c in row.keys() if c.startswith("default_pred_n")])

            scores = {}
            scores['protac_smiles'] = row['protac_smiles']
            scores['label_smiles'] = row['label_smiles']
            for i in range(num_return_sequences):
                pred_smiles = row[f'default_pred_n{i}']

                curr_scores = score_prediction(
                    protac_smiles=row['protac_smiles'],
                    label_smiles=row['label_smiles'], 
                    pred_smiles=pred_smiles,
                    compute_graph_metrics=True,
                    graph_edit_kwargs={'timeout': 0.1}
                )
                metric_names = list(curr_scores.keys()) # Save them for later
                curr_scores = {f"pred_n{i}_{k}": v for k, v in curr_scores.items()}

                # If the prediction is already correct, skip the fixing of the
                # prediction and copy the scores to the final scores.
                if pred_smiles == row['label_smiles']:
                    curr_scores.update({f"fixed_{k}": v for k, v in curr_scores.items()})
                    curr_scores[f"fixed_pred_n{i}_fixed"] = True
                    scores.update(curr_scores)
                    continue

                curr_scores[f"fixed_pred_n{i}_fixed"] = True
                fixed_smiles = fix_prediction(row['protac_smiles'], pred_smiles)

                if fixed_smiles is None:
                    curr_scores.update({f"fixed_pred_n{i}_{k}": curr_scores[f"pred_n{i}_{k}"] for k in metric_names})
                    curr_scores[f"fixed_pred_n{i}_fixed"] = False

                elif fixed_smiles == pred_smiles:
                    curr_scores.update({f"fixed_pred_n{i}_{k}": curr_scores[f"pred_n{i}_{k}"] for k in metric_names})

                else:
                    fixed_scores = score_prediction(
                        protac_smiles=row['protac_smiles'],
                        label_smiles=row['label_smiles'], 
                        pred_smiles=fixed_smiles,
                        compute_graph_metrics=True,
                        graph_edit_kwargs={'timeout': 0.1}
                    )
                    curr_scores.update({f"fixed_pred_n{i}_{k}": v for k, v in fixed_scores.items()})

                scores.update(curr_scores)
            
            # Remove unused score metrics: tanimoto
            scores = {k: v for k, v in scores.items() if 'tanimoto' not in k}
            return scores

        # Use the map function to apply the scoring function to each row
        scores = ds.map(
            score_multiple_predictions,
            num_proc=num_proc,
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
            graph_edit_kwargs={'timeout': 1.0}
        )
        score['protac_smiles'] = row['protac_smiles']
        score['label_smiles'] = row['label_smiles']
        score['pred_smiles'] = row[config]
        score['preds_config'] = config
        scores.append(score)

    # Save scored predictions
    pd.DataFrame(scores).to_csv("logs/scores.csv", index=False)


if __name__ == "__main__":
    # Setup the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes to use for scoring predictions")
    args = parser.parse_args()
    main(
        num_proc=args.num_proc,
    )