"""
PROTAC Splitter Web Application

This script provides a web interface for splitting PROTAC molecules into their
constituent parts: E3 ligase binder, linker, and protein-of-interest (POI) 
ligand (warhead).

The app uses the protac_splitter library to perform the splitting and offers
two main modes of operation:
1. Single SMILES processing
2. Batch processing via CSV file upload

Users can select which models to use:
- XGBoost model (default): Fast graph-based edge classification model
- Transformer model: More accurate but slower deep learning model
- If neither is selected, a rule-based splitting algorithm is used

Author: Stefano Ribes
Date: 2025-06
"""

import logging
import tempfile
from pathlib import Path
from typing import Union

from PIL import Image
import gradio as gr
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from protac_splitter import split_protac
from protac_splitter.display_utils import get_mapped_protac_img

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

def process_single_smiles(protac_smiles: str, use_transformer: bool = False, use_xgboost: bool = True, beam_size: int = 5) -> tuple:
    """
    Process a single SMILES string and generate PROTAC fragment predictions
    
    Args:
        protac_smiles: The SMILES string of the PROTAC molecule
        use_transformer: Whether to use the transformer model for prediction
        use_xgboost: Whether to use the XGBoost model for prediction
        
    Returns:
        Tuple containing input image, output images, SMILES texts and status message
    """
    if not protac_smiles:
        raise gr.Error("Please provide a valid PROTAC SMILES string.", duration=5)

    try:
        results = split_protac(
            protac_smiles,
            use_transformer=use_transformer,  
            use_xgboost=use_xgboost,      
            fix_predictions=True,   # Always apply fixes to predictions
            beam_size=beam_size,    # Use beam search width for Transformer model
            verbose=1
        )
    except Exception as e:
        exception_message = str(e)
        if exception_message.startswith("Invalid PROTAC SMILES"):
            raise gr.Error("The input SMILES string is not valid (couldn't be parsed by RDKit).", duration=5)
        else:
            raise gr.Error(f"An error occurred while processing the input SMILES: {exception_message}", duration=10)

    valid_molecules = []
    pred_key = f'default_pred_n0'
    valid_molecules.append(results[pred_key])

    # Generate images and corresponding SMILES text
    images = []
    smiles_texts = []
    input_mol = Chem.MolFromSmiles(protac_smiles)
    
    if input_mol is not None:
        input_img = Draw.MolToImage(input_mol, legend="", size=(1000, 200))
    else:
        input_img = Image.new('RGB', (1000, 1000))
    
    splits = {}
    for smiles in results[pred_key].split("."):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            if "[*:1]" in smiles and "[*:2]" in smiles:
                legend = "Linker"
                splits['linker'] = smiles
            elif "[*:1]" in smiles:
                legend = "Warhead"
                splits['poi'] = smiles
            elif "[*:2]" in smiles:
                legend = "E3 Ligase Ligand"
                splits['e3'] = smiles

            img = Draw.MolToImage(mol, legend="", size=(1000, 1000))
            images.append(img)
            smiles_texts.append(f"{legend}: {smiles}")
    smiles_texts = "\n".join(smiles_texts)
    
    use_svg = False
    input_img = get_mapped_protac_img(
        protac_smiles=protac_smiles,
        poi_smiles=splits.get('poi', ''),
        linker_smiles=splits.get('linker', ''),
        e3_smiles=splits.get('e3', ''),
        w=1000,
        h=500,
        legend=None,
        useSVG=use_svg,
    )

    if use_svg:
        input_img = save_svg_to_tempfile(input_img)
        logging.debug(f"Returning processed image path: {input_img}")
    
    return input_img, list(images), smiles_texts

def process_csv(
        file: gr.File,
        smiles_col: str,
        use_transformer: bool = False,
        use_xgboost: bool = True,
        beam_size: int = 5,
        batch_size: int = 4,
        num_proc: int = 2,
        # NOTE: `pr` is a progress tracker, it is used to track the progress but
        # it is not used in this function. Do not remove it.
        pr: gr.Progress = gr.Progress(track_tqdm=True),
) -> Path:
    """
    Process a CSV file containing PROTAC SMILES
    
    Args:
        file: Uploaded CSV file
        smiles_col: Name of the column containing SMILES strings
        use_transformer: Whether to use the transformer model for prediction
        use_xgboost: Whether to use the XGBoost model for prediction
        
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
            use_transformer=use_transformer,
            use_xgboost=use_xgboost,
            protac_smiles_col=smiles_col,
            fix_predictions=True,
            batch_size=batch_size,
            num_proc=num_proc,
            beam_size=beam_size,  # Use beam search width for Transformer model
            verbose=1
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
    with gr.Blocks() as demo:        
        header = """# PROTAC-Splitter Web Application

Upload a CSV file or enter a single SMILES string to predict PROTAC substructures.

Warheads and E3 ligase ligands connections to the linker are marked with dummy atoms, _i.e._, attachment points, as follows:

- Warhead: `[*:1]`
- E3 Ligase ligand: `[*:2]`

"""
        gr.Markdown(header)
        
        # Model selection section - common to both tabs
        model_selection = """## Model Selection
        
You can choose which model to use for splitting PROTAC molecules:

- **XGBoost model** (default): Fast graph-based edge classification model
- **Transformer model**: More accurate but slower deep learning model
- If both are selected, the Transformer model will be used first, then if it fails, the XGBoost model will be used.
- If no model is selected, splitting will be done using graph-based heuristics, with no AI model involved.

For fast splitting, we reccommend using the XGBoost model only, which is fast and efficient for most cases. The Transformer model might be more accurate but it is slower, especially for processing large CSV files.
"""
        gr.Markdown(model_selection)
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    use_xgboost = gr.Checkbox(label="Use XGBoost model", value=True)
                    use_transformer = gr.Checkbox(label="Use Transformer model", value=False)
        
        # Performance configuration section
        performance_configs = """### Performance Configurations

Change the following parameters to optimize performance based on your machine's capabilities. Particularly useful when processing large CSV files or when using the Transformer model.
For single SMILES processing, the default values should work well in most cases.
"""
        gr.Markdown(performance_configs)
        with gr.Column(scale=1):
            # Add a num_proc input
            with gr.Row():
                num_proc = gr.Number(
                    label="Number of Processes",
                    value=2,
                    minimum=1,
                    maximum=8,
                    step=1,
                    info="Number of processes to use for parallel processing. Higher values may improve performance but require more memory."
                )

            # Add a number input for beam_size if Transformer model is selected
            with gr.Row():
                # Only show beam size input if Transformer model is selected
                beam_size = gr.Number(
                    label="Beam Search Width",
                    value=5,
                    minimum=1,
                    maximum=10,
                    step=1,
                    info="Width of the beam search for the Transformer model. Higher values may improve accuracy but increase processing time.",
                    visible=use_transformer.value  # Initially hidden, will be shown if Transformer is selected
                )
                # Add a dynamic visibility condition to show/hide beam_size based on Transformer model selection
                use_transformer.change(
                    lambda x: gr.update(visible=x),
                    inputs=[use_transformer],
                    outputs=[beam_size]
                )

            # Add a batch size input for Transformer model if selected
            with gr.Row():
                batch_size = gr.Number(
                    label="Batch Size",
                    value=4,
                    minimum=1,
                    maximum=64,
                    step=1,
                    info="Batch size for processing. Higher values may improve performance, especially on GPU machines, but require more memory.",
                    visible=use_transformer.value  # Initially hidden, will be shown if Transformer is selected
                )
                use_transformer.change(
                    lambda x: gr.update(visible=x),
                    inputs=[use_transformer],
                    outputs=[batch_size]
                )

        # Single SMILES Input tab
        gr.Markdown("## Specify Inputs")
        with gr.Tab("Single SMILES Input"):
            # Input area
            smiles_input = gr.Textbox(
                label="Enter SMILES String", 
                placeholder="E.g., CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O",
                # value="CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O",
            )

            submit_smiles = gr.Button("Process SMILES")

            # Output area
            smiles_input_image = gr.Image(label="Input PROTAC", type="filepath")  # Use None to allow SVG input
            smiles_output_images = gr.Gallery(label="Valid Splits", columns=3)
            smiles_output_texts = gr.Textbox(label="SMILES of the Splits", interactive=False, lines=3)

            # Connect the button click event to the processing function
            submit_smiles.click(
                process_single_smiles, 
                inputs=[smiles_input, use_transformer, use_xgboost, beam_size], 
                outputs=[smiles_input_image, smiles_output_images, smiles_output_texts]
            )

        # CSV file processing tab
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
                inputs=[file_input, smiles_column, use_transformer, use_xgboost, beam_size, batch_size, num_proc],
                outputs=[download_output]
            )
            
            csv_notes = f"""**Note:** The output CSV will contain the following columns:

- `{smiles_column}`: The original PROTAC SMILES string
- `default_pred_n0`: The predicted SMILES strings for the splits
- `model_name`: The model used for the prediction
"""
            gr.Markdown(csv_notes)

    return demo

# Create the Gradio interface
# NOTE: `demo` must be a global variable, so to make the Gradio’s hot-reload system work.
# NOTE: Launch the app with `gradio scripts/protac_splitter_app.py` to develop it.
demo = create_interface()

if __name__ == "__main__":
    # Set logging level to DEBUG for detailed output
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    demo.launch()
