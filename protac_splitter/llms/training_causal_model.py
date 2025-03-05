
import os
from typing import Dict, Any
import torch
from transformers import TrainerCallback
from trl import SFTTrainer
from rdkit import Chem

from .data_utils import load_tokenized_dataset
from .evaluation import decode_and_get_metrics
from .hf_utils import create_hf_repository, delete_hf_repository, repo_exists
from .model_utils import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU if available

# Placeholder for a scoring function that evaluates the generated SMILES
def score_function(smiles1, predicted_smiles):
    """ Evaluates the generated SMILES sequence based on validity. """
    mol = Chem.MolFromSmiles(predicted_smiles)
    return 1 if mol else 0  # Returns 1 if valid, 0 if invalid

# Custom Trainer subclass to integrate SMILES evaluation
class CustomSFTTrainer(SFTTrainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # Generate predictions
        predictions = self.predict(eval_dataset)
        generated_texts = self.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

        total_score = 0
        total_samples = len(generated_texts)

        for i, example in enumerate(eval_dataset):
            input_text = example["text"]  # Full input: "Smiles1 Smiles2.Smiles3.Smiles4"
            smiles1 = input_text.split(" ")[0]  # Extract Smiles1 (the prompt)

            # Remove the prompt from the generated text to get the predicted completion
            predicted_completion = generated_texts[i].removeprefix(smiles1).strip()

            # Compute custom score
            score = score_function(smiles1, predicted_completion)
            total_score += score

        # Compute average score
        average_score = total_score / total_samples if total_samples > 0 else 0

        # Log metrics
        metrics = {f"{metric_key_prefix}_average_score": average_score}
        self.log(metrics)

        return metrics

def train():
    """ Main training function """
    model = get_model()  # Load the model
    tokenizer = model.tokenizer  # Get tokenizer from model
    
    # Load dataset
    dataset = load_tokenized_dataset()

    # Training arguments
    training_args = {
        "output_dir": "./trained_model",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "logging_steps": 100,
        "save_steps": 500,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 5e-5,
        "save_total_limit": 2,
    }

    # Initialize custom trainer
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    train()
