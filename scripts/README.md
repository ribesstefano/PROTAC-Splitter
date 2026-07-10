# Useful Scripts using the PROTAC-Splitter

This directory contains scripts that can be used to interact with the PROTAC-Splitter package.

## Table of Contents 📜

- [Generate Finetuning Dataset](#generate-finetuning-dataset)
- [Finetuning Model](#finetuning-model)
- [Collect LLMs Predictions](#collect-llms-predictions)
- [Score Predictions](#score-predictions)
- [QC Dataset](#qc-dataset)
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

## QC Dataset

To flag suspect rows in a PROTAC-Splitter dataset or prediction CSV for manual review — foreign/non-PROTAC molecules, unstable or synthesis-artifact substructures, implausible linkers, and splits that are structurally valid but likely cut at the wrong bonds — you can use the [`scripts/qc_dataset.py`](../scripts/qc_dataset.py) script. The actual checks live in `protac_splitter.data.curation.dataset_qc`. Nothing is deleted or auto-corrected: every row gets a `n_flags` count and a `review_reasons` string, and the output CSV is sorted worst-first for triage.

Please run `python scripts/qc_dataset.py --help` for more information on the arguments. Example of usage:

```bash
python scripts/qc_dataset.py --input_csv="tack_smiles_split.csv" --n_jobs=8
```

By default the output is written next to the input as `<input_csv stem>.qc.csv`. The `--run_heuristic_agreement` and `--run_xgboost_confidence` flags (on by default) re-split every molecule with the betweenness-centrality heuristic and score it with the XGBoost edge classifier, respectively — both add runtime but are the main split-correctness signals. Set `--limit N` for a quick smoke test on the first N rows before running the full dataset.

### Output columns

**Structural validity**

| Column | Meaning |
|---|---|
| `valid_protac_smiles` | Original SMILES parses in RDKit. |
| `pred_valid` | The `e3.linker.poi` prediction string parses. |
| `has_three_substructures` | Prediction has exactly 2 dots (3 fragments). |
| `has_all_attachment_points` | Exactly two `[*:1]` and two `[*:2]` present. |
| `reassembly_ok` | Gluing the 3 fragments back together reproduces the original molecule exactly. |

**Fragment size / connectivity**

| Column | Meaning |
|---|---|
| `e3_mw`, `poi_mw` | Molecular weight of that fragment (dummy atom stripped, capped with H). |
| `e3_heavy_atoms`, `poi_heavy_atoms` | Heavy-atom count of that fragment. |
| `e3_disconnected`, `poi_disconnected` | True if the fragment itself contains a `.` (multiple pieces) — usually a parsing artifact. |
| `flag_e3_out_of_range`, `flag_poi_out_of_range` | MW falls outside a plausible range for that role (E3: 150–700 Da, POI: 120–900 Da). |

**Linker topology**

| Column | Meaning |
|---|---|
| `linker_heavy_atoms_between` | Heavy atoms strictly between the two attachment points along the shortest path. |
| `linker_branch_points` | Non-ring atoms in the linker with 3+ connections (real tree-branching, not just a substituted ring like piperazine). |
| `linker_ring_count` | Rings present in the linker fragment. |
| `flag_linker_too_short` | ≤1 heavy atom between attachment points — suspiciously little separation between E3 and POI. |
| `flag_linker_branchy` | 2+ branch points — linkers are normally near-linear chains. |

**Chemical plausibility**

| Column | Meaning |
|---|---|
| `brenk_hits` | Semicolon-list of RDKit BRENK unstable/reactive substructure matches on the *intact* molecule (informational — includes categories that don't gate the flag). |
| `flag_unstable` | True if any BRENK hit falls outside the allowlisted categories (excludes `phthalimide`/`Aliphatic_long_chain`/`aniline`, which are common and legitimate in real PROTACs). |
| `leaving_group_hits` | Semicolon-list of matched synthesis-artifact SMARTS (boronates, silyl ethers, Boc/Cbz/Fmoc, tosylate/mesylate, azide, diazo, alkyl halide). |
| `flag_leaving_group` | True if any leaving-group pattern matched anywhere in the molecule. |

**Split correctness — known-ligand identity**

| Column | Meaning |
|---|---|
| `e3_sim_to_known_e3` | Max Tanimoto similarity of the predicted E3 fragment to the curated E3-ligand reference set. |
| `e3_sim_to_known_wh` | Same fragment's similarity to the warhead reference set (for detecting swaps). |
| `poi_sim_to_known_wh`, `poi_sim_to_known_e3` | Same, mirrored for the POI fragment. |
| `flag_e3_low_similarity`, `flag_poi_low_similarity` | Similarity to its own expected reference set is below threshold (0.2 default) — fragment doesn't look like a known E3 ligand / warhead. |
| `flag_role_swap_suspected` | The E3 fragment looks more like a warhead *and* the POI fragment looks more like an E3 ligand than the reverse — likely the two roles got swapped. |

**Split correctness — cross-method agreement**

| Column | Meaning |
|---|---|
| `heuristic_e3` / `heuristic_linker` / `heuristic_poi` | What the betweenness-centrality heuristic (independent of the model that made the given prediction) splits this same molecule into. |
| `heuristic_min_similarity` | The lowest per-role Tanimoto similarity between the given prediction and the heuristic's split (worst of the three fragments). |
| `flag_method_disagreement` | `heuristic_min_similarity` below threshold (0.6 default) — the two methods cut the molecule substantially differently. |

**Split correctness — model confidence**

| Column | Meaning |
|---|---|
| `xgb_top1_proba` | XGBoost edge classifier's probability for its own top-ranked cut bond on this molecule. |
| `xgb_margin` | Gap between the top-1 and top-2 candidate cut-bond probabilities — a small margin means the model itself was choosing between two similarly-plausible bonds. |
| `flag_low_confidence` | `xgb_margin` below threshold (0.15 default). |

**Rollup**

| Column | Meaning |
|---|---|
| `flag_structural` | Shorthand for "any of the 5 structural checks failed." |
| `n_flags` | Count of all triggered `flag_*` columns — the output CSV is sorted by this, worst first. |
| `review_reasons` | Semicolon-joined names of every triggered flag, for quick eyeballing without scanning the boolean columns individually. |

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
