#!/bin/bash

# Path to the local dataset for fine-tuning, e.g., generated via the "get_finetuning_dataset.py" script.
ds_name="data/finetuning_dataset"

# Dataset configuration. Change this to use a different dataset configuration (e.g., "n20", "n50", etc.).
# Configurations are subfolders in the dataset directory.
ds_config="n10"

# Name of the model to be trained. Modify this to set a different model name or configuration.
# NOTE: The final directory structure will look like this: <output_dir>/<model_name>/<checkpoint>
model_name="PROTAC-Splitter-EncoderDecoder-finetuned-${ds_config}"

# Batch size for training on each device. Increase this for larger GPUs or decrease for memory constraints.
per_device_train_batch_size=1

# Batch size for evaluation on each device. Adjust this based on evaluation requirements and GPU memory.
per_device_eval_batch_size=1

# Number of gradient accumulation steps. Modify this to simulate a larger batch size.
gradient_accumulation_steps=1

# Directory where the trained model and checkpoints will be saved. Change this to specify a different output location.
# NOTE: The final directory structure will look like this: <output_dir>/<model_name>/<checkpoint>
output_dir="models/"

# Path to the checkpoint (directory) to resume training from. Update this to use a different checkpoint or start fresh.
# NOTE: The final directory structure will look like this: <output_dir>/<model_name>/<checkpoint>
checkpoint="models/PROTAC-Splitter-EncoderDecoder-lr_reduce-rand-smiles/last-checkpoint"

# Whether to delete existing repositories before training. Set to "false" to retain existing repositories.
delete_repos=true

# Whether to randomize SMILES strings during training. Set to "true" to enable randomization.
randomize_smiles=false

# Probability of randomizing SMILES strings. Adjust this value if randomization is enabled.
randomize_smiles_prob=0.0

# Maximum number of training steps. Increase or decrease this based on the desired training duration.
max_steps=1000

# Number of training epochs. Set to a positive value for epoch-based training or leave as -1 for step-based training.
# NOTE: Either max_steps or num_train_epochs should be set, not both, i.e., one of them must be -1.
num_train_epochs=-1

python scripts/train_transformer_model.py \
    --model_id="${model_name}" \
    --ds_name="${ds_name}" \
    --output_dir="${output_dir}" \
    --ds_config="${ds_config}" \
    --batch_size_tokenizer=1024 \
    --per_device_train_batch_size="${per_device_train_batch_size}" \
    --per_device_eval_batch_size="${per_device_eval_batch_size}" \
    --gradient_accumulation_steps="${gradient_accumulation_steps}" \
    --max_steps="${max_steps}" \
    --num_train_epochs="${num_train_epochs}" \
    --resume_from_checkpoint="${checkpoint}" \
    --num_proc_map=6 \
    --delete_local_repo_if_exists="${delete_repos}" \
    --randomize_smiles="${randomize_smiles}" \
    --randomize_smiles_prob="${randomize_smiles_prob}"