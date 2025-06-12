# Useful Scripts using the PROTAC-Splitter

This directory contains scripts that can be used to interact with the PROTAC-Splitter package.

## Table of Contents 📜

- [Generate Finetuning Dataset](#generate-finetuning-dataset)
- [Finetuning Model](#finetuning-model)
- [Collect LLMs Predictions](#collect-llms-predictions)
- [Score Predictions](#score-predictions)
- [Plotting Scores](#plotting-scores)
- [Plotting the Chemical Space](#plotting-the-chemical-space)
- [PROTAC-Splitter App](#protac-splitter-app)

## Generate Finetuning Dataset

To cluster 10, 20, 50, and 100 representative PROTACs from your dataset, you can use the [`scripts/get_finetuning_dataset.py`](../scripts/get_finetuning_dataset.py) script. This script will take care of the following steps:
1. **Load the dataset**: The script will load the dataset from a CSV file. The CSV file should contain one column for each SMILES of the PROTAC and its three ligands.
2. **Cluster the PROTACs**: The script will cluster the PROTACs via K-means clustering.
3. **Generate the finetuning dataset**: The script will then generate the finetuning dataset (readeable by HuggingFace) under the specified directory.

Please run `python scripts/get_finetuning_dataset.py --help` for more information on the arguments.

Example of usage:

```bash
python scripts/get_finetuning_dataset.py --filename_held_out_df=data/processed/mapped_protacs_with_functional_groups.csv --ds_root=data/finetuning_dataset
```

## Finetuning Model

To finetune the model on your own dataset, you can use the [`scripts/finetune_model.sh`](../scripts/finetune_model.sh) script. Please modify the script to setup the correct paths to your dataset and the model you want to finetune, as well as the resulting training directory.

Example of usage:

```bash
bash scripts/finetune_model.sh
```

## Collect LLMs Predictions

To collect the predictions from the finetuned model, you can use the [`scripts/collect_predictions.py`](../scripts/collect_predictions.py) script.

Please run `python scripts/collect_predictions.py --help` for more information on the arguments. Example of usage:

```bash
python scripts/collect_llm_predictions.py --model_name="PROTAC-Splitter-Finetuned" --dataset_dir="data/finetuning_dataset" --dataset_config="n10" --dataset_test_split="test" --log_dir="logs"
```

In the example above, the model will be run to predict on the `data/finetuning_dataset/n10/test` dataset, _i.e._, at rootdir `data/finetuning_dataset/`, configuration `n10` (10 clustered PROTACs) and test split `test`. The model will be loaded from the `PROTAC-Splitter-Finetuned` directory.

## Collect Graph-Based Predictions

To collect the predictions from the graph-based model, you can use the [`scripts/collect_graph_predictions.py`](../scripts/collect_graph_predictions.py) script.

Please run `python scripts/collect_graph_predictions.py --help` for more information on the arguments. Example of usage with a pre-trained model:

```bash
python scripts/collect_graph_predictions.py --input_csv="your/awesome/PROTACs/to/split.csv" --output_csv="logs/example_output.csv" --classifier_model="models/edge_classifier_graph_features_bin.joblib" --n_jobs=4 --batch_size 256 --smiles_column="protac_smiles" --labels_column="label_smiles"
```

## Score Predictions

To score the predictions, you can use the [`scripts/score_predictions.py`](../scripts/score_predictions.py) script.

Please run `python scripts/score_predictions.py --help` for more information on the arguments. Example of usage:

```bash
python scripts/score_predictions.py --log_dir="logs" --num_proc=8
```

## Plotting Scores

To plot the predictions, you can use the [`scripts/plot_predictions.py`](../scripts/plotting.py) script.
Please run `python scripts/plotting.py --help` for more information on the arguments. Example of usage:

```bash
python scripts/plotting.py --score_file="logs/PROTAC-Splitter-Model-v2-scores.csv" --img_dir="images"
```

## Plotting the Chemical Space

To plot the chemical space of the PROTACs, you can use the [`scripts/plot_chemical_space.py`](../scripts/plot_chemical_space.py) script.
Please run `python scripts/plot_chemical_space.py --help` for more information on the arguments. Example of usage:

```bash
python scripts/plot_chemical_space.py --protac_db_path=data/raw/PROTAC-DB-v3.csv --protac_pedia_path=data/raw/PROTAC-Pedia.csv --num_proc=8 --num_proc_fp_gen=8 --internal_data_path=path/to/interna/data.csv
```

Notice that the fingerprint generation can take a while, so the first run is recommended to run this script on a machine with multiple cores.

## PROTAC-Splitter App

We also provide a simple Gradio app to interact with the PROTAC-Splitter model. The app can be run using the [`scripts/protac_splitter_app.py`](../scripts/protac_splitter_app.py) script.
The app will be usually available at `http://localhost:7860` but please double-check your terminal for the precise address.

If running on a remote server, one could run the script above, then on your local machine, open a terminal and run the following command:

```bash
ssh -L 7860:127.0.0.1:7860 username@remote_serve
```

After running the above command, you can open a web browser on your local machine and navigate to: `http://127.0.0.1:7860`

**NOTE**: By default the model will try to run on a GPU, if available. If not available, the model will run on CPU, but can be very slow even for predicting one single PROTAC.
