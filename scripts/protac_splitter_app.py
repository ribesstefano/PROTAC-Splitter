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
XGBoost, heuristic, Transformer, a combination of these, or the QC-gated
"adaptive" strategy that escalates through several of them.

Author: Stefano Ribes
Date: 2025-06
"""

import faulthandler
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Union

# Diagnostic: if any request stalls this long, dump every thread's exact Python
# stack to stderr (visible in the Space's log stream) and keep dumping every 20s
# while it's still stuck. Cheap, always-on, and turns "it hangs" into "it's stuck
# at this exact line" the next time it happens.
faulthandler.enable()
faulthandler.dump_traceback_later(20, repeat=True, exit=False, file=sys.stderr)

# Import protac_splitter before gradio/pandas/rdkit below: it sets thread-count env
# vars for native math libraries (see protac_splitter/config.py), which some of them
# only read once, at first load — importing it after gradio (which pulls in numpy
# transitively) would be too late.
from protac_splitter import split_protac
from protac_splitter.config import get_cache_dir
from protac_splitter.evaluation import split_prediction

from PIL import Image
import gradio as gr
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# HF Spaces sets SPACE_ID automatically; cap parallelism on the (limited) free tier.
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None
MAX_NUM_PROC = 2 if IS_HF_SPACE else 8

# Filename must match `_XGBOOST_MODEL_FILENAME` in protac_splitter/protac_splitter.py.
# If a copy of the model is bundled alongside this script (as it is on the HF Space,
# to avoid depending on a runtime download from Zenodo), seed the cache with it before
# any request can trigger a download.
_BUNDLED_XGBOOST_MODEL = Path(__file__).with_name("PROTAC-Splitter-XGBoost.joblib")
if _BUNDLED_XGBOOST_MODEL.exists():
    _cached_model_path = get_cache_dir() / _BUNDLED_XGBOOST_MODEL.name
    if not _cached_model_path.exists():
        _cached_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_BUNDLED_XGBOOST_MODEL, _cached_model_path)
        logging.info(f"Seeded XGBoost model cache from bundled file → {_cached_model_path}")

MODEL_CHOICES = [
    ("Heuristic → XGBoost (recommended)", "heuristic->xgboost"),
    ("XGBoost (fast)", "xgboost"),
    ("Heuristic (no model)", "heuristic"),
    ("XGBoost → Heuristic", "xgboost->heuristic"),
    ("XGBoost + Heuristic (best of both)", "xgboost+heuristic"),
    ("Transformer", "transformer"),
    ("Transformer → XGBoost", "transformer->xgboost"),
    ("Adaptive (QC-gated escalation, best quality)", "adaptive"),
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
        adaptive_use_xgboost: bool = True,
        adaptive_use_transformer: bool = False,
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
        adaptive_use_xgboost: Whether the XGBoost stage runs on molecules the
            heuristic grid left flagged, only used when `model == "adaptive"`
        adaptive_use_transformer: Whether the Transformer stage runs on molecules
            still flagged after XGBoost, only used when `model == "adaptive"`

    Returns:
        Tuple containing input image, output images, SMILES texts, substructure
        dataframe, and a status message with the winning model/QC info
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
            adaptive_use_xgboost=adaptive_use_xgboost,
            adaptive_use_transformer=adaptive_use_transformer,
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

    # `n_flags` / `review_reasons` / `heuristic_params` are only present when
    # model="adaptive" (see evaluation.score_split / count_flags).
    info_lines = [f"Model used: {results.get('model_name')}"]
    if "n_flags" in results:
        reasons = results.get("review_reasons") or "none"
        params = results.get("heuristic_params")
        info_lines.append(
            f"Remaining QC flags: {results['n_flags']} ({reasons})"
            + (f" [winning heuristic params: {params}]" if params else "")
        )
    info_text = "\n".join(info_lines)

    return input_img, images, smiles_texts, smiles_df, info_text

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
        adaptive_use_xgboost: bool = True,
        adaptive_use_transformer: bool = False,
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
        adaptive_use_xgboost: Whether the XGBoost stage runs on molecules the
            heuristic grid left flagged, only used when `model == "adaptive"`
        adaptive_use_transformer: Whether the Transformer stage runs on molecules
            still flagged after XGBoost, only used when `model == "adaptive"`

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
            adaptive_use_xgboost=adaptive_use_xgboost,
            adaptive_use_transformer=adaptive_use_transformer,
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
- **Adaptive**: QC-gated escalation, not just fallback-on-failure — tries a heuristic parameter grid first,
  then XGBoost, then (if enabled below) the Transformer, keeping whichever candidate scores best on
  automated plausibility checks. Slower than a single strategy, but generally the highest-quality split;
  also reports which method/parameters won and any remaining review flags. See **Adaptive Settings** below.
""")
        with gr.Row():
            with gr.Column(scale=2):
                model = gr.Radio(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Splitting strategy",
                )

        heuristic_settings_label = gr.Markdown(
            "### Heuristic Settings\n\nOnly used when the heuristic algorithm is part of the selected splitting strategy above (including **Adaptive**, whose parameter grid is seeded with these values).",
            visible="heuristic" in DEFAULT_MODEL or DEFAULT_MODEL == "adaptive",
        )
        with gr.Row(visible="heuristic" in DEFAULT_MODEL or DEFAULT_MODEL == "adaptive") as heuristic_settings_row:
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

        adaptive_settings_label = gr.Markdown(
            "### Adaptive Settings\n\nOnly used when the **Adaptive** splitting strategy is selected above.",
            visible=DEFAULT_MODEL == "adaptive",
        )
        with gr.Row(visible=DEFAULT_MODEL == "adaptive") as adaptive_settings_row:
            adaptive_use_xgboost = gr.Checkbox(
                label="Use XGBoost stage",
                value=True,
                info="Run the XGBoost edge classifier on molecules the heuristic grid left flagged.",
            )
            adaptive_use_transformer = gr.Checkbox(
                label="Use Transformer stage",
                value=False,
                info="Run the Transformer model on molecules still flagged after XGBoost. Requires the [transformer] extra; slow on CPU.",
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
                    visible="transformer" in DEFAULT_MODEL or DEFAULT_MODEL == "adaptive",
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
                    visible="transformer" in DEFAULT_MODEL or DEFAULT_MODEL == "adaptive",
                )

            # Show/hide the Transformer-only, heuristic-only, and adaptive-only options based on the selected strategy
            model.change(
                lambda m: (
                    gr.update(visible="transformer" in m or m == "adaptive"),
                    gr.update(visible="transformer" in m or m == "adaptive"),
                    gr.update(visible="heuristic" in m or m == "adaptive"),
                    gr.update(visible="heuristic" in m or m == "adaptive"),
                    gr.update(visible=m == "adaptive"),
                    gr.update(visible=m == "adaptive"),
                ),
                inputs=[model],
                outputs=[
                    beam_size, batch_size, heuristic_settings_label, heuristic_settings_row,
                    adaptive_settings_label, adaptive_settings_row,
                ],
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
            smiles_output_info = gr.Textbox(
                label="Split Info",
                interactive=False,
                lines=2,
                info="Winning model, and (for the Adaptive strategy) remaining QC flags / winning heuristic params.",
            )

            # Add this Examples component
            gr.Examples(
                examples=[
                    # SMILES, model, beam_size, betweenness_threshold, use_capacity_weight,
                    # betweenness_approx_frac, adaptive_use_xgboost, adaptive_use_transformer
                    ["CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O", "heuristic->xgboost", 5, 0.4, False, None, True, False],
                    ["Cc1nnc2n1-c1sc(C#Cc3cnn(-c4cccc5c4C(=O)N(C4CCC(=O)NC4=O)C5=O)c3)c(Cc3ccccc3)c1COC2", "heuristic->xgboost", 5, 0.4, False, None, True, False],
                    ["c1ccccc1CCC1CCCC1", "heuristic", 5, 0.4, False, None, True, False],
                    ["O=C(NCCOCCOCCN1CCCC1)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O", "heuristic", 5, 0.4, False, None, True, False],
                    ["CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O", "adaptive", 5, 0.4, False, None, True, False],
                ],
                inputs=[
                    smiles_input, model, beam_size, betweenness_threshold, use_capacity_weight,
                    betweenness_approx_frac, adaptive_use_xgboost, adaptive_use_transformer,
                ],
                outputs=[smiles_input_image, smiles_output_images, smiles_output_texts, smiles_output_df, smiles_output_info],
                fn=process_single_smiles,
                cache_examples=True,
            )

            # Connect the button click event to the processing function.
            # Own concurrency lane, sized above the CSV lane: single-SMILES calls are
            # cheap, so several can run at once without starving each other.
            submit_smiles.click(
                process_single_smiles,
                inputs=[
                    smiles_input, model, beam_size, betweenness_threshold, use_capacity_weight,
                    betweenness_approx_frac, adaptive_use_xgboost, adaptive_use_transformer,
                ],
                outputs=[smiles_input_image, smiles_output_images, smiles_output_texts, smiles_output_df, smiles_output_info],
                concurrency_limit=4,
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

            # Connect the button click event to the processing function.
            # Own concurrency lane, capped at 1: batch jobs already use num_proc worker
            # processes internally, so running several batches at once on a 2-core Space
            # would oversubscribe the CPU rather than speed anything up.
            submit_csv.click(
                process_csv,
                inputs=[
                    file_input, smiles_column, model, beam_size, batch_size, num_proc,
                    betweenness_threshold, use_capacity_weight, betweenness_approx_frac,
                    adaptive_use_xgboost, adaptive_use_transformer,
                ],
                outputs=[download_output],
                concurrency_limit=1,
            )

            gr.Markdown(f"""**Note:** The output CSV will contain the following columns:

- `smiles_column`: The original PROTAC SMILES string
- `default_pred_n0`: The predicted SMILES strings for the splits
- `model_name`: The model used for the prediction
- With the **Adaptive** strategy, three extra columns: `heuristic_params` (which grid point won,
  when `model_name == "Heuristic"`, else empty), `n_flags`, and `review_reasons`
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
# Bound the total queue backlog so a burst of requests degrades predictably (early
# "queue full" errors) instead of piling up unboundedly on a 2-core Space.
demo.queue(max_size=32)

if __name__ == "__main__":
    # Set logging level to DEBUG for detailed output
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    demo.launch()
