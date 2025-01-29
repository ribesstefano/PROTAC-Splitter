from io import BytesIO
from typing import List, Dict, Any

from PIL import Image
import gradio as gr
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from protac_splitter import split_protac

def process_single_smiles(protac_smiles: str, model_path: str):
    try:
        results = split_protac(protac_smiles, model_name=model_path, return_check_reassembly=True)
    except Exception as e:
        exception_message = str(e)
        if exception_message.startswith("Invalid PROTAC SMILES"):
            raise gr.Error("The input SMILES string is not valid (couldn't be parsed by RDKit).", duration=5)
        else:
            raise gr.Error(f"An error occurred while processing the input SMILES: {exception_message}", duration=10)

    valid_molecules = []
    invalid_count = 0
    duplicates_count = 0
    max_splits = 5
    
    for i in range(max_splits):
        pred_key = f'default_pred_n{i}'
        check_key = f'reassembly_correct_n{i}'

        if check_key in results:
            if results[check_key]:
                if results[pred_key] not in valid_molecules:
                    valid_molecules.append(results[pred_key])
                else:
                    duplicates_count += 1
            else:
                invalid_count += 1
    
    # Generate images and corresponding SMILES text
    images = []
    smiles_texts = []
    input_mol = Chem.MolFromSmiles(protac_smiles)
    if input_mol:
        input_img = Draw.MolToImage(input_mol, legend="Input PROTAC", size=(1000, 1000))
    else:
        input_img = Image.new('RGB', (1000, 1000))
    
    for i, smiles in enumerate(valid_molecules):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, legend=f"Predition n.{i+1}", size=(1000, 1000))
            images.append(img)
            smiles_texts.append(smiles)
    
    message = f"{invalid_count} out of {max_splits} predicted splits failed the reassembly check. Number of duplicate splits: {duplicates_count} (only one duplicate is showed)."

    smiles_texts = "\n".join([f"Predition n.{i+1}: {s}" for i, s in enumerate(smiles_texts)])
    
    return input_img, images, smiles_texts, message

def process_csv(file, smiles_col: str, model_path: str):
    df = pd.read_csv(file.name)
    if smiles_col not in df.columns:
        # Use Gradio's error message instead of raising an exception
        raise gr.Error(f"Column \"{smiles_col}\" is not in the provided CSV file.", duration=5)

    try:
        results = split_protac(df,
            model_name=model_path,
            protac_smiles_col=smiles_col,
            return_check_reassembly=False,
            fix_predictions=False,
        )
    except Exception as e:
        exception_message = str(e)
        if exception_message.startswith("Invalid PROTAC SMILES"):
            raise gr.Error("One or more of the input SMILES are not valid (couldn't be parsed by RDKit).", duration=5)
        else:
            raise gr.Error(f"An error occurred while processing: {exception_message}", duration=10)


    # # TODO: Add a flag button to fix the predictions
    # fixed_results = split_protac(df,
    #     model_name=model_path,
    #     protac_smiles_col=smiles_col,
    #     return_check_reassembly=False,
    #     fix_predictions=True,
    # )
    
    output_df = pd.DataFrame(results)
    # fixed_df = pd.DataFrame(fixed_results)

    # Add a column with the input SMILES
    output_df.insert(0, "protac_smiles", df[smiles_col])
    # fixed_df.insert(0, "protac_smiles", df[smiles_col])

    # Merge the results and fixed_results dictionaries
    # output_df = pd.merge(output_df, fixed_df, on="protac_smiles", suffixes=('', '_fixed'))
    
    output_file = "split_results.csv"
    output_df.to_csv(output_file, index=False)
    
    return output_file

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# PROTAC Splitter")
        gr.Markdown("Upload a CSV file or enter a single SMILES string to predict PROTAC fragments.\n\nWarheads and E3 ligands connections (bonding) to the linker are marked with dummy atoms, i.e., attachment points. For the warhead, we have \"[*:1]\", whereas we have \"[*:2]\" for the E3 ligand.")
        
        # model_path = gr.Textbox(label="Local Model Path", placeholder="Enter the local model directory", value="/mimer/NOBACKUP/groups/naiss2023-6-290/stefano/models/PROTAC-Splitter-Trial-11")
        model_path = gr.Textbox(label="Local Model Path", placeholder="Enter the local model directory, e.g., /download/directory/PROTAC-Splitter-Trial-11")
        
        with gr.Tab("Single SMILES Input"):
            # smiles_input = gr.Textbox(label="Enter SMILES String", value="CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O")
            smiles_input = gr.Textbox(label="Enter SMILES String", placeholder="E.g., CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O")
            submit_smiles = gr.Button("Process SMILES")
            smiles_input_image = gr.Image(label="Input PROTAC")
            smiles_output_images = gr.Gallery(label="Valid Splits", columns=1)
            smiles_output_texts = gr.Textbox(label="SMILES of the Splits", interactive=False, lines=5)
            smiles_output_message = gr.Textbox(label="Reassembly Status", interactive=False)

            submit_smiles.click(
                process_single_smiles, 
                inputs=[smiles_input, model_path], 
                outputs=[smiles_input_image, smiles_output_images, smiles_output_texts, smiles_output_message]
            )
        
        with gr.Tab("Upload CSV"):
            file_input = gr.File(label="Upload CSV File")
            smiles_column = gr.Textbox(label="Column Name for SMILES", placeholder="e.g., \"PROTAC SMILES\"")
            submit_csv = gr.Button("Process CSV")
            download_output = gr.File(label="Download Predictions")
            
            submit_csv.click(
                process_csv, 
                inputs=[file_input, smiles_column, model_path], 
                outputs=[download_output]
            )
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
