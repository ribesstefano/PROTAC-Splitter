from pathlib import Path

import pandas as pd
from tqdm import tqdm

from protac_splitter.evaluation import score_prediction

def main():
    # Read predictions CSV
    predictions_path = Path("logs/predictions.csv")
    if not predictions_path.exists():
        raise FileNotFoundError("predictions.csv not found in logs directory")
        
    df = pd.read_csv(predictions_path)

    # Remove rows in which the label_smiles has more than two dots
    df = df[~df['label_smiles'].str.contains("\.\[Cl-\]\.")]
    
    preds_configs = [c for c in df.columns if c.startswith("preds_")]

    # Apply scoring function to each row
    scores = []
    for i, row in tqdm(df.iterrows(), desc="Scoring predictions", total=len(df)):
        for config in preds_configs:
            score = score_prediction(
                protac_smiles=row['protac_smiles'],
                label_smiles=row['label_smiles'], 
                pred_smiles=row[config],
                compute_graph_metrics=True,
                graph_edit_kwargs={'timeout': 0.1}
            )
            score['protac_smiles'] = row['protac_smiles']
            score['label_smiles'] = row['label_smiles']
            score['preds_config'] = config
            scores.append(score)

    # Save scored predictions
    pd.DataFrame(scores).to_csv("logs/scores.csv", index=False)
    
if __name__ == "__main__":
    main()