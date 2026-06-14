# PROTAC-Splitter: A Machine Learning Framework for Automated Identification of PROTAC Substructures

This repository contains the program code to split PROTAC molecules into their three constituent substructures: the **E3 ligand**, the **linker**, and the **POI warhead**.

A Gradio app is available to split PROTAC molecules and visualize the results: [https://huggingface.co/spaces/ailab-bio/PROTAC-Splitter-App](https://huggingface.co/spaces/ailab-bio/PROTAC-Splitter-App).

## Table of Contents 📜

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Python API](#python-api)
  - [Command-line interface](#command-line-interface)
  - [Gradio app](#gradio-app)
- [Splitting strategies](#splitting-strategies)
- [Data availability](#data-availability)
- [Contributing](#contributing)
- [License](#license)
- [Reference](#reference)

---

## Installation 🛠️

Requires Python 3.10+. Always use a virtual environment.

### pip (recommended)

```bash
# Core package — includes XGBoost and heuristic splitting (no GPU required)
pip install git+https://github.com/ribesstefano/PROTAC-Splitter.git

# With Transformer model support (adds PyTorch; ~2 GB download)
pip install "git+https://github.com/ribesstefano/PROTAC-Splitter.git[transformer]"

# With Gradio app + plotting dependencies
pip install "git+https://github.com/ribesstefano/PROTAC-Splitter.git[scripts]"

# Full development install (all extras + pytest)
pip install "git+https://github.com/ribesstefano/PROTAC-Splitter.git[dev]"
```

### From source with uv

```bash
git clone https://github.com/ribesstefano/PROTAC-Splitter.git
cd PROTAC-Splitter
pip install uv
uv sync --extra dev        # install all extras into .venv
source .venv/bin/activate
```

> **Pretrained models are downloaded automatically on first use.**  
> The XGBoost model (~17 MB) is cached to `~/.cache/protac_splitter/` on the first call to `split_protac()`.  
> The Transformer model is cached by HuggingFace `transformers` in `~/.cache/huggingface/`.

---

## Configuration ⚙️

Create a `.env` file in your working directory (or copy `.env.example`) to override defaults:

```bash
cp .env.example .env
```

```ini
# .env
# Directory where pretrained models are cached (default: ~/.cache/protac_splitter)
PROTAC_SPLITTER_CACHE_DIR=~/.cache/protac_splitter

# HuggingFace token — only needed to access private Hub models
HF_TOKEN=
```

Environment variables are loaded automatically via `python-dotenv` when the package is imported.

---

## Usage 🚀

### Python API 🐍

```python
from protac_splitter import split_protac

# --- Single SMILES string ---
protac = (
    "CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O"
    "Cc1nnc2n1-c1sc(C#Cc3cnn(-c4cccc5c4C(=O)N(C4CCC(=O)NC4=O)C5=O)c3)c(Cc3ccccc3)c1COC2"
)
result = split_protac(protac)
print(result)
# {'text': '...', 'default_pred_n0': 'e3_smiles.linker_smiles.poi_smiles', 'model_name': 'XGBoost'}

# --- List of SMILES ---
results = split_protac([protac, protac])
for r in results:
    print(r["default_pred_n0"])

# --- pandas DataFrame ---
import pandas as pd
df = pd.read_csv("protacs.csv") # must have a column for SMILES strings
result_df = split_protac(df, protac_smiles_col="PROTAC SMILES")
print(result_df[["PROTAC SMILES", "default_pred_n0", "model_name"]].head())
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` (via `split_protac`) | — | See [Splitting strategies](#splitting-strategies) |
| `use_xgboost` | `True` | Use XGBoost graph edge classifier |
| `use_transformer` | `False` | Use seq2seq Transformer model |
| `fix_predictions` | `True` | Apply cheminformatics post-processing |
| `beam_size` | `5` | Beam-search width (Transformer only) |
| `betweenness_threshold` | `0.4` | Split-point sensitivity (heuristic / XGBoost fallback) |
| `use_capacity_weight` | `False` | Weight edges by bond order in betweenness centrality |
| `num_proc` | `1` | Parallel worker processes |

### Command-line interface 🖥️

After installation the `protac-splitter` command is available:

```bash
# Split a single SMILES with the default XGBoost model
protac-splitter --smiles "CC(C)(C)S(=O)(=O)c1cc2c..."

# Use the heuristic algorithm (no model download)
protac-splitter --smiles "..." --model heuristic

# Tune the heuristic threshold
protac-splitter --smiles "..." --model heuristic --betweenness-threshold 0.5

# Use bond-capacity weighting in betweenness centrality
protac-splitter --smiles "..." --model heuristic --use-capacity-weight

# Transformer model (requires [transformer] extra)
protac-splitter --smiles "..." --model transformer

# Transformer with XGBoost fallback
protac-splitter --smiles "..." --model transformer+xgboost

# Batch processing from CSV
protac-splitter --input-csv protacs.csv --smiles-col "SMILES" \
                --output-csv results.csv --model xgboost

# Machine-readable CSV output
protac-splitter --smiles "..." --model heuristic --output-format csv

# Show all options
protac-splitter --help
```

**Available models (`--model`):**

| Value | Description |
|---|---|
| `xgboost` | XGBoost graph edge classifier — default, no GPU required |
| `heuristic` | Betweenness-centrality graph algorithm — no model download |
| `transformer` | Seq2seq Transformer (requires `[transformer]` extra) |
| `transformer+xgboost` | Transformer with XGBoost fallback on failed reassemblies |

### Gradio app 🌐

```bash
gradio scripts/protac_splitter_app.py
# Open http://localhost:7860
```

---

## Splitting strategies 🧠

PROTAC-Splitter supports three strategies, selectable via `--model` (CLI) or via `use_transformer` / `use_xgboost` flags (Python API):

1. **XGBoost** (`--model xgboost`) — graph edge classifier trained on synthetic PROTACs. Recommended for batch processing. Model downloaded automatically (~17 MB) on first use.

2. **Transformer** (`--model transformer`) — seq2seq encoder–decoder model hosted on HuggingFace (`ailab-bio/PROTAC-Splitter`). Requires the `[transformer]` extra and a GPU for reasonable throughput.

3. **Heuristic** (`--model heuristic`) — betweenness-centrality algorithm. No model download. Useful for quick exploration or air-gapped environments.

4. **Transformer + XGBoost** (`--model transformer+xgboost`) — runs the Transformer first and falls back to XGBoost for predictions that fail cheminformatics reassembly.

### Output format

All strategies return predictions in a dot-separated SMILES format:

```
e3_smiles.linker_smiles.poi_smiles
```

Attachment points are encoded as `[*:1]` (POI side) and `[*:2]` (E3 side).

---

## Data availability 📊

Curated datasets and trained models are available on Zenodo:  
[https://doi.org/10.5281/zenodo.15797309](https://doi.org/10.5281/zenodo.15797309)

The XGBoost model is downloaded automatically to `$PROTAC_SPLITTER_CACHE_DIR` on first use. No manual download step is needed.

---

## Contributing 🤝

We welcome contributions! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

---

## License 📄

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Reference 📚

If you find this work useful, please consider citing it:

```bibtex
@article{ribes2026protac,
  title={{PROTAC-Splitter: a machine learning framework for automated identification of PROTAC substructures: S. Ribes et al.}},
  author={Ribes, Stefano and Zhang, Ranxuan and Cropsal, T{\'e}lio and K{\"a}llberg, Anders and Tyrchan, Christian and Nittinger, Eva and Mercado, Roc{\'\i}o},
  journal={Journal of Cheminformatics},
  volume={18},
  number={1},
  pages={30},
  year={2026},
  publisher={Springer}
}
```
