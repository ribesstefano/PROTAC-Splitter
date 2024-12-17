# PROTAC Splitter

This repository contains a program to split PROTAC molecules into their substructures.

## Quickstart

Install the required dependencies and activate the relative environment:

```bash
conda env create -f environment.yml
conda activate env-protac-splitter
```

Run the following command for starting training the model:

```bash
mkdir -p models
hub_token="my-unforgettable-token"
organization="my-awesome-organization"

python main.py train_model \
    "PROTAC-Splitter_untied_80-20-split_with-sampling" \
    "ailab-bio/PROTAC-Substructures" \
    --organization=${organization} \
    --ds_config="80-20-split" \
    --tokenizer="seyonec/ChemBERTa-zinc-base-v1" \
    --pretrained_encoder="seyonec/ChemBERTa-zinc-base-v1" \
    --pretrained_decoder="seyonec/ChemBERTa-zinc-base-v1" \
    --tie_encoder_decoder=false \
    --output_dir="/models" \
    --batch_size=256 \
    --max_steps=2000 \
    --num_train_epochs=-1 \
    --hub_token="${hub_token}" \
    --delete_repo_first=false
```

In general, refer to the help message for more information about the command line arguments:

```bash
python main.py --help
```

## Data Preparation

The train and test datasets are assembled in the notbook: [notebooks/data_curation.ipynb](notebooks/data_curation.ipynb).

Raw CSV data are expected to be placed in the `data/raw` directory.


## Code Cleaning

- Started organizing Anders' curation code into a set of files in the `protac_splitter/gnn` directory.
- Clustering code needs some polishing and refactoring, i.e., handling plotting and visualizations
- There is a huge function for generating the train/val/test splits, but I still don't understand how it differs from other functions in the clustering file...