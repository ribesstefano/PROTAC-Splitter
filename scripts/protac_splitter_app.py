"""
PROTAC Splitter Web Application

This script provides a web interface for splitting PROTAC molecules into their
constituent parts: E3 ligase binder, linker, and protein-of-interest (POI)
ligand (warhead).

The app uses the protac_splitter library to perform the splitting and offers
two main modes of operation:
1. Single SMILES processing
2. Batch processing via CSV file upload

Users choose a splitting strategy (see `model` in `protac_splitter.split_protac`):
XGBoost, heuristic, Transformer, or a combination of these.

Author: Stefano Ribes
Date: 2025-06
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Union

from PIL import Image
import gradio as gr
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from protac_splitter import split_protac
from protac_splitter.evaluation import split_prediction

# HF Spaces sets SPACE_ID automatically; cap parallelism on the (limited) free tier.
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None
MAX_NUM_PROC = 2 if IS_HF_SPACE else 8

MODEL_CHOICES = [
    ("Heuristic → XGBoost (recommended)", "heuristic->xgboost"),
    ("XGBoost (fast)", "xgboost"),
    ("Heuristic (no model)", "heuristic"),
    ("XGBoost → Heuristic", "xgboost->heuristic"),
    ("XGBoost + Heuristic (best of both)", "xgboost+heuristic"),
    ("Transformer", "transformer"),
    ("Transformer → XGBoost", "transformer->xgboost"),
]
DEFAULT_MODEL = "heuristic->xgboost"

SUBSTRUCTURE_LEGENDS = {"e3": "E3 Ligase Ligand", "linker": "Linker", "poi": "Warhead (POI)"}


def save_svg_to_tempfile(svg_string: str, suffix: str = ".svg") -> Union[str, Path]:
    """
    Write an SVG string to a temporary file and return its filesystem path.
    """
    # Create a named temporary file that persists after closing
    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    logging.debug(f"Saving SVG to temporary file: {tmp_file.name}")
    try:
        tmp_file.write(svg_string)
        tmp_file.flush()
        return Path(tmp_file.name)
    finally:
        tmp_file.close()

def process_single_smiles(
        protac_smiles: str,
        model: str = DEFAULT_MODEL,
        beam_size: int = 5,
        betweenness_threshold: float = 0.4,
        use_capacity_weight: bool = False,
        betweenness_approx_frac: float = None,
) -> tuple:
    """
    Process a single SMILES string and generate PROTAC fragment predictions

    Args:
        protac_smiles: The SMILES string of the PROTAC molecule
        model: Splitting strategy to use (see `protac_splitter.split_protac`)
        beam_size: Beam search width, only used by strategies involving the Transformer
        betweenness_threshold: Betweenness-centrality cut-off, only used when the
            heuristic strategy is part of `model`
        use_capacity_weight: Weight graph edges by bond capacity, heuristic only
        betweenness_approx_frac: Fraction of nodes sampled for approximate betweenness
            centrality, heuristic only. Leave empty for exact computation.

    Returns:
        Tuple containing input image, output images, SMILES texts and status message
    """
    if not protac_smiles:
        raise gr.Error("Please provide a valid PROTAC SMILES string.", duration=5)

    try:
        results = split_protac(
            protac_smiles,
            model=model,
            fix_predictions=True,   # Always apply fixes to predictions
            beam_size=beam_size,    # Use beam search width for Transformer model
            betweenness_threshold=betweenness_threshold,
            use_capacity_weight=use_capacity_weight,
            betweenness_approx_frac=betweenness_approx_frac,
            verbose=1,
        )
    except Exception as e:
        exception_message = str(e)
        if exception_message.startswith("Invalid PROTAC SMILES"):
            raise gr.Error("The input SMILES string is not valid (couldn't be parsed by RDKit).", duration=5)
        else:
            raise gr.Error(f"An error occurred while processing the input SMILES: {exception_message}", duration=10)

    input_mol = Chem.MolFromSmiles(protac_smiles)
    if input_mol is not None:
        input_img = Draw.MolToImage(input_mol, legend="", size=(1000, 200))
    else:
        input_img = Image.new("RGB", (1000, 1000))

    splits = split_prediction(results.get("default_pred_n0"))

    images = []
    for key in ("e3", "linker", "poi"):
        smiles = splits.get(key)
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            images.append((Draw.MolToImage(mol, legend="", size=(1000, 1000)), SUBSTRUCTURE_LEGENDS[key]))

    if any(splits.get(key) is None for key in ("e3", "linker", "poi")):
        smiles_texts = "Prediction failed for one or more substructures. Please try again with a different splitting strategy."
    else:
        smiles_texts = ".".join(splits[key] for key in ("e3", "linker", "poi"))

    smiles_df = pd.DataFrame({
        "Substructure": [SUBSTRUCTURE_LEGENDS["e3"], SUBSTRUCTURE_LEGENDS["linker"], SUBSTRUCTURE_LEGENDS["poi"]],
        "SMILES": [splits.get("e3") or "FAILED", splits.get("linker") or "FAILED", splits.get("poi") or "FAILED"],
    })

    return input_img, images, smiles_texts, smiles_df

def process_csv(
        file: gr.File,
        smiles_col: str,
        model: str = DEFAULT_MODEL,
        beam_size: int = 5,
        batch_size: int = 4,
        num_proc: int = 2,
        betweenness_threshold: float = 0.4,
        use_capacity_weight: bool = False,
        betweenness_approx_frac: float = None,
        # NOTE: `pr` is a progress tracker, it is used to track the progress but
        # it is not used in this function. Do not remove it.
        pr: gr.Progress = gr.Progress(track_tqdm=True),
) -> Path:
    """
    Process a CSV file containing PROTAC SMILES

    Args:
        file: Uploaded CSV file
        smiles_col: Name of the column containing SMILES strings
        model: Splitting strategy to use (see `protac_splitter.split_protac`)
        beam_size: Beam search width, only used by strategies involving the Transformer
        betweenness_threshold: Betweenness-centrality cut-off, only used when the
            heuristic strategy is part of `model`
        use_capacity_weight: Weight graph edges by bond capacity, heuristic only
        betweenness_approx_frac: Fraction of nodes sampled for approximate betweenness
            centrality, heuristic only. Leave empty for exact computation.

    Returns:
        Path to output CSV file with predictions
    """
    df = pd.read_csv(file.name)
    if smiles_col not in df.columns:
        # Use Gradio's error message instead of raising an exception
        raise gr.Error(f"Column \"{smiles_col}\" is not in the provided CSV file.", duration=5)

    try:
        results = split_protac(
            df,
            model=model,
            protac_smiles_col=smiles_col,
            fix_predictions=True,
            batch_size=batch_size,
            num_proc=min(num_proc, MAX_NUM_PROC),
            beam_size=beam_size,  # Use beam search width for Transformer model
            betweenness_threshold=betweenness_threshold,
            use_capacity_weight=use_capacity_weight,
            betweenness_approx_frac=betweenness_approx_frac,
            verbose=1,
        )
    except Exception as e:
        exception_message = str(e)
        if exception_message.startswith("Invalid PROTAC SMILES"):
            raise gr.Error("One or more of the input SMILES are not valid (couldn't be parsed by RDKit).", duration=5)
        else:
            raise gr.Error(f"An error occurred while processing: {exception_message}", duration=10)

    output_df = pd.DataFrame(results)

    # Create a temporary output file
    output_file = str(Path(tempfile.gettempdir()) / "split_preds.csv")
    logging.debug(f"Saving predictions to temporary file: {output_file}")
    output_df.to_csv(output_file, index=False)
    logging.debug(f"Output DataFrame saved to: {output_file}")

    return output_file

def create_interface():
    """
    Create and return the Gradio interface for the PROTAC Splitter app

    The interface includes two tabs:
    1. Single SMILES Input - For processing individual PROTAC SMILES
    2. CSV Upload - For batch processing of multiple PROTAC SMILES

    Returns:
        gr.Blocks: The Gradio interface
    """
    css = """
h1 {
    text-align: center;
    display:block;
}
"""
    with gr.Blocks(css=css) as demo:
        # ----------------------------------------------------------------------
        # Application title and description
        # ----------------------------------------------------------------------
        gr.Markdown("""# ✂️ PROTAC-Splitter Web Application ✂️

Upload a CSV file or enter a single SMILES string to predict PROTAC substructures.

Warheads (protein-of-interest ligands) and E3 ligase ligands connections to the linker are marked with dummy atoms, _i.e._, attachment points, as follows:

- Warhead (POI): `[*:1]`
- E3 Ligase ligand: `[*:2]`
""")

        # ----------------------------------------------------------------------
        # Model selection section - common to both tabs
        # ----------------------------------------------------------------------
        gr.Markdown(f"""## Splitting Strategy

You can choose which strategy to use for splitting PROTAC molecules:

- **Heuristic → XGBoost** (default, recommended): runs the fast graph heuristic first and only falls back to
  XGBoost for predictions that fail to reassemble. Robust, fast, and accurate for general use.
- **XGBoost**: fast graph-based edge classification model.
- **Heuristic**: betweenness-centrality algorithm, no model download needed.
- **XGBoost → Heuristic** / **XGBoost + Heuristic**: alternative combinations of the two graph-based strategies.
- **Transformer** / **Transformer → XGBoost**: often more accurate, but a much slower deep learning model.
  {"Disabled on this Hugging Face Space (CPU-only, too slow for interactive use)." if IS_HF_SPACE else "Runs on CPU, so it is slower, especially for large CSV files."}
""")
        with gr.Row():
            with gr.Column(scale=2):
                model = gr.Radio(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Splitting strategy",
                )

        heuristic_settings_label = gr.Markdown(
            "### Heuristic Settings\n\nOnly used when the heuristic algorithm is part of the selected splitting strategy above.",
            visible="heuristic" in DEFAULT_MODEL,
        )
        with gr.Row(visible="heuristic" in DEFAULT_MODEL) as heuristic_settings_row:
            betweenness_threshold = gr.Slider(
                label="Betweenness Threshold",
                value=0.4,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                info="Betweenness-centrality cut-off for the heuristic algorithm. Higher values are more conservative.",
            )
            use_capacity_weight = gr.Checkbox(
                label="Use Capacity Weight",
                value=False,
                info="Weight graph edges by bond capacity when computing betweenness centrality.",
            )
            betweenness_approx_frac = gr.Number(
                label="Betweenness Approx. Fraction (optional)",
                value=None,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                info="Fraction of nodes to sample for approximate betweenness centrality. Leave empty for exact computation.",
            )

        # ----------------------------------------------------------------------
        # Performance configuration section
        # ----------------------------------------------------------------------
        gr.Markdown("""### Performance Configurations

Change the following parameters to optimize performance based on your machine's capabilities. Particularly useful when processing large CSV files or when using the Transformer model.
For single SMILES processing, the default values should work well in most cases.
""")
        with gr.Column(scale=1):
            # Add a num_proc input
            with gr.Row():
                num_proc = gr.Number(
                    label="Number of Processes",
                    value=min(2, MAX_NUM_PROC),
                    minimum=1,
                    maximum=MAX_NUM_PROC,
                    step=1,
                    info=(
                        "Number of processes to use for parallel processing. Higher values may improve performance "
                        "but require more memory."
                        + (f" (Capped to {MAX_NUM_PROC} on this Hugging Face Space)" if IS_HF_SPACE else "")
                    ),
                )

            # Add a number input for beam_size, only relevant for Transformer-based strategies
            with gr.Row():
                beam_size = gr.Number(
                    label="Beam Search Width",
                    value=5,
                    minimum=1,
                    maximum=10,
                    step=1,
                    info="Width of the beam search for the Transformer model. Higher values may improve accuracy but increase processing time.",
                    visible="transformer" in DEFAULT_MODEL,
                )

            # Add a batch size input, only relevant for Transformer-based strategies
            with gr.Row():
                batch_size = gr.Number(
                    label="Batch Size",
                    value=1,
                    minimum=1,
                    maximum=64,
                    step=1,
                    info="Batch size for processing. Higher values may improve performance, especially on GPU machines, but require more memory.",
                    visible="transformer" in DEFAULT_MODEL,
                )

            # Show/hide the Transformer-only and heuristic-only options based on the selected strategy
            model.change(
                lambda m: (
                    gr.update(visible="transformer" in m),
                    gr.update(visible="transformer" in m),
                    gr.update(visible="heuristic" in m),
                    gr.update(visible="heuristic" in m),
                ),
                inputs=[model],
                outputs=[beam_size, batch_size, heuristic_settings_label, heuristic_settings_row],
            )

        # ----------------------------------------------------------------------
        # Single SMILES Input tab
        # ----------------------------------------------------------------------
        gr.Markdown("""## Specify Inputs

**Disclaimer**: The input SMILES is checked for validity before processing. There is no check on whether the SMILES is a PROTAC-like molecule or not.
For example, attempting to split the SMILES `c1ccccc` (benzene) with the XGBoost or heuristic strategies will return an error, as ring bonds are ignored for splitting.
On the other end, `c1ccccc1CCC1CCCC1` will return a plausible split, even though it is not a PROTAC molecule.
""")
        with gr.Tab("Single SMILES Input"):
            # Input area
            # NOTE: A challenging SMILES to test the app is: CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O
            smiles_input = gr.Textbox(
                label="Enter SMILES String",
                placeholder="E.g., CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O",
            )
            submit_smiles = gr.Button("Process SMILES")

            # Output area
            smiles_input_image = gr.Image(label="Input PROTAC")
            smiles_output_images = gr.Gallery(
                label="Predicted Splits",
                columns=3,
            )
            smiles_output_df = gr.DataFrame(
                label="Substructure Predictions",
                interactive=False,
                headers=["Substructure", "SMILES"],
                show_copy_button=True,
            )
            smiles_output_texts = gr.Textbox(
                label="SMILES of the Splits",
                interactive=False,
                lines=1,
                show_copy_button=True,
            )

            # Add this Examples component
            gr.Examples(
                examples=[
                    # SMILES, model, beam_size, betweenness_threshold, use_capacity_weight, betweenness_approx_frac
                    ["CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O", "heuristic->xgboost", 5, 0.4, False, None],
                    ["Cc1nnc2n1-c1sc(C#Cc3cnn(-c4cccc5c4C(=O)N(C4CCC(=O)NC4=O)C5=O)c3)c(Cc3ccccc3)c1COC2", "heuristic->xgboost", 5, 0.4, False, None],
                    ["c1ccccc1CCC1CCCC1", "heuristic", 5, 0.4, False, None],
                    ["O=C(NCCOCCOCCN1CCCC1)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O", "heuristic", 5, 0.4, False, None],
                ],
                inputs=[smiles_input, model, beam_size, betweenness_threshold, use_capacity_weight, betweenness_approx_frac],
                outputs=[smiles_input_image, smiles_output_images, smiles_output_texts, smiles_output_df],
                fn=process_single_smiles,
                cache_examples=True,
            )

            # Connect the button click event to the processing function
            submit_smiles.click(
                process_single_smiles,
                inputs=[smiles_input, model, beam_size, betweenness_threshold, use_capacity_weight, betweenness_approx_frac],
                outputs=[smiles_input_image, smiles_output_images, smiles_output_texts, smiles_output_df]
            )

        # ----------------------------------------------------------------------
        # CSV file processing tab
        # ----------------------------------------------------------------------
        with gr.Tab("Upload CSV"):
            # File upload area
            file_input = gr.File(label="Upload CSV File")
            smiles_column = gr.Textbox(
                label="Column Name for PROTAC SMILES",
                placeholder="E.g., \"PROTAC SMILES\"",
                # value="PROTAC SMILES",
            )
            submit_csv = gr.Button("Process CSV")

            # Output file download area
            download_output = gr.File(label="Download Predictions")

            # Connect the button click event to the processing function
            submit_csv.click(
                process_csv,
                inputs=[
                    file_input, smiles_column, model, beam_size, batch_size, num_proc,
                    betweenness_threshold, use_capacity_weight, betweenness_approx_frac,
                ],
                outputs=[download_output]
            )

            gr.Markdown(f"""**Note:** The output CSV will contain the following columns:

- `smiles_column`: The original PROTAC SMILES string
- `default_pred_n0`: The predicted SMILES strings for the splits
- `model_name`: The model used for the prediction
""")

        # ----------------------------------------------------------------------
        # Citation
        # ----------------------------------------------------------------------
        gr.Markdown("""---
If you find this work useful, please consider citing it:

```bibtex
@article{ribes2026protac,
  title={{PROTAC-Splitter: a machine learning framework for automated identification of PROTAC substructures: S. Ribes et al.}},
  author={Ribes, Stefano and Zhang, Ranxuan and Cropsal, T{\\'e}lio and K{\\"a}llberg, Anders and Tyrchan, Christian and Nittinger, Eva and Mercado, Roc{\\'\\i}o},
  journal={Journal of Cheminformatics},
  volume={18},
  number={1},
  pages={30},
  year={2026},
  publisher={Springer}
}
```
""")

    return demo

# Create the Gradio interface
# NOTE: `demo` must be a global variable, so to make the Gradio's hot-reload system work.
# NOTE: Launch the app with `gradio scripts/protac_splitter_app.py` to develop it.
demo = create_interface()

if __name__ == "__main__":
    # Set logging level to DEBUG for detailed output
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    demo.launch()
