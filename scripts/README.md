# Useful Scripts using the PROTAC-Splitter

This directory contains scripts that can be used to interact with the PROTAC-Splitter package.

## Table of Contents 📜

- [Generate Finetuning Dataset](#generate-finetuning-dataset)
- [protac_splitter_app](#protac_splitter_app)

## Generate Finetuning Dataset

```bash
python scripts/get_finetuning_dataset.py --filename_held_out_df=data/processed/mapped_protacs_with_functional_groups.csv --ds_root=data/finetuning_dataset
```

## Collect LLMs Predictions

## Score Predictions

## PROTAC-Splitter App

<!-- ```bash
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /mimer/NOBACKUP/groups/naiss2023-6-290/stefano/envs/env-protac-splitter/bin/activate
export PYTHONPATH=$PYTHONPATH:`pwd`/protac_splitter
export PYTHONPATH=/mimer/NOBACKUP/groups/naiss2023-6-290/stefano/envs/env-protac-splitter/lib/python3.11/site-packages:$PYTHONPATH
PYTHONNOUSERSITE=1 python -m scripts.protac_splitter_app
``` -->