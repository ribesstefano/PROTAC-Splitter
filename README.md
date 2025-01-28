# PROTAC-Splitter

This repository contains a program to split PROTAC molecules into their substructures.

<!-- Add some emojies to the subsections -->
## Table of Contents 📜

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation 🛠️

The package was tested in Python 3.10.8.

For using the code under the [scripts](scripts) directory in this repository, run the following commands:

```bash
git clone https://github.com/ribesstefano/PROTAC-Splitter.git
cd PROTAC-Splitter
pip install -r requirements.txt
pip install -r scripts/requirements.txt

# Add the package to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`/protac_splitter
```

Alternatively, you can install the package using pip:

```bash
pip install git+https://github.com/ribesstefano/PROTAC-Splitter.git
```

## Usage 🚀

To use the package, please refer to the function `split_protac` in the [protac_splitter/protac_splitter](protac_splitter/protac_splitter) module.

Here is an example of how to use the function:

```python
from protac_splitter import split_protac

# Split a PROTAC molecule
protac_smiles = "CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O"

# Split the PROTAC molecule
ligands = split_protac(
    protac_smiles,
    model_name="change_to_local_model_path_if_required",
    hf_token="your_awsome_hf_token_or_os.environ['HF_TOKEN']",
)
print(ligands)

# One can also feed a DataFrame to the function
df = pd.read_csv("my/local/file.csv")
split_df = split_protac(
    df,          
    model_name="change_to_local_model_path_if_required",
    hf_token="your_awsome_hf_token_or_os.environ['HF_TOKEN']",
    protac_smiles_col="PROTAC SMILES",
)
print(split_df.head())
```

Alternatively, you can use the Gradio app at [scripts/protac_splitter_app.py](scripts/protac_splitter_app.py) have a GUI to split PROTAC molecules.

```sh
PYTHONNOUSERSITE=1 python -m script.protac_splitter_app
```

Until the repository is private, please download the model locally from this Google Drive link: https://drive.google.com/file/d/18hq62csehlmQlzfQoAAgmiV_vMT0AcP0/view?usp=share_link

After unzipping, set the `model_name` argument to the path of the directory downloaded model. At this point, there is no need to set the `hf_token` argument.

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.