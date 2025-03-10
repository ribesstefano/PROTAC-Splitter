import os
from pathlib import Path
import logging
import argparse

import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from protac_splitter.evaluation import score_prediction
from protac_splitter.protac_splitter import fix_prediction
from protac_splitter.chemoinformatics import canonize

def main(
    num_proc: int = 16,
    skip_is_log_exists: bool = False,
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
        print('-' * 80)
        print(f"Model name: {predictions_path.stem}")
        print('-' * 80)
        print(f"Scoring predictions in {predictions_path}")
        scores_path = predictions_path.with_name(predictions_path.stem.replace("preds", "scores") + ".csv")

        if skip_is_log_exists and os.path.exists(scores_path):
            print(f"Scores already exist for {predictions_path.stem}, skipping...")
            continue
        print(f"Scores will be saved in {scores_path}")

        # Read predictions CSV
        df = pd.read_csv(predictions_path)

        # Remove rows in which the label_smiles has more than two dots
        df = df[~df['label_smiles'].str.contains("\.\[Cl-\]\.")]

        # Canonize the SMILES strings
        df['protac_smiles'] = df['protac_smiles'].apply(canonize)
        df['label_smiles'] = df['label_smiles'].apply(canonize)

        # Convert the pandas DataFrame to a Hugging Face Dataset
        ds = Dataset.from_pandas(df, preserve_index=False)
        
        def score_multiple_predictions(row: dict):
            # Columns: protac_smiles,label_smiles,default_pred_n0,greedy_pred_n0,contrastive_search_pred_n0,multinomial_sampling_pred_n0,beam_search_decoding_pred_n0,beam_search_decoding_pred_n1,beam_search_decoding_pred_n2,beam_search_decoding_pred_n3,beam_search_decoding_pred_n4,beam_search_multinomial_sampling_pred_n0,beam_search_multinomial_sampling_pred_n1,beam_search_multinomial_sampling_pred_n2,beam_search_multinomial_sampling_pred_n3,beam_search_multinomial_sampling_pred_n4,diverse_beam_search_decoding_pred_n0,diverse_beam_search_decoding_pred_n1,diverse_beam_search_decoding_pred_n2,diverse_beam_search_decoding_pred_n3,diverse_beam_search_decoding_pred_n4,model_name
            scores = {}
            scores['protac_smiles'] = protac_smiles = row['protac_smiles']
            scores['label_smiles'] = label_smiles = row['label_smiles']
            
            for pred_name, pred_smiles in row.items():
                if pred_name in ['protac_smiles', 'label_smiles', 'model_name'] or '_n' not in pred_name:
                    continue
                
                curr_scores = score_prediction(
                    protac_smiles=protac_smiles,
                    label_smiles=label_smiles,
                    pred_smiles=pred_smiles,
                    compute_graph_metrics=True,
                    graph_edit_kwargs={'timeout': 0.1}
                )
                metric_names = list(curr_scores.keys()) # Save them for later
                curr_scores = {f"{pred_name}_{metric}": v for metric, v in curr_scores.items()}
                
                if pred_smiles == label_smiles:
                    # If the prediction is already correct, skip fixing the
                    # prediction and copy the scores to the "fixed" scores.
                    curr_scores.update({f"{pred_name_metric}_fixed": v for pred_name_metric, v in curr_scores.items()})
                    curr_scores[f"{pred_name}_is_fixed"] = True
                    scores.update(curr_scores)
                    continue
                
                fixed_smiles = fix_prediction(protac_smiles, pred_smiles)
                curr_scores.update({f"{pred_name}_{metric}_fixed": curr_scores[f"{pred_name}_{metric}"] for metric in metric_names})

                if fixed_smiles is None:
                    curr_scores[f"{pred_name}_is_fixed"] = False
                elif fixed_smiles == pred_smiles:
                    curr_scores[f"{pred_name}_is_fixed"] = True
                else:
                    fixed_scores = score_prediction(
                        protac_smiles=protac_smiles,
                        label_smiles=label_smiles,
                        pred_smiles=fixed_smiles,
                        compute_graph_metrics=True,
                        graph_edit_kwargs={'timeout': 0.1}
                    )
                    curr_scores.update({f"{pred_name}_{metric}_fixed": v for metric, v in fixed_scores.items()})
                    curr_scores[f"{pred_name}_is_fixed"] = True

                curr_scores['model_name'] = row['model_name']
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
        print(f"Scores saved in {scores_path}")
        print()


if __name__ == "__main__":
    # Setup the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes to use for scoring predictions")
    parser.add_argument("--skip_is_log_exists", action="store_true", help="Skip scoring if the scores file already exists")
    args = parser.parse_args()
    main(
        num_proc=args.num_proc,
        skip_is_log_exists=args.skip_is_log_exists,
    )