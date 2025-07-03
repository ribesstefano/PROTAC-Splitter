# PROTAC-Splitter

This repository contains the program code to split PROTAC molecules into their constituent substructures.

A Gradio app is available to split PROTAC molecules and visualize the results at this link: [https://huggingface.co/spaces/ailab-bio/PROTAC-Splitter-App](https://huggingface.co/spaces/ailab-bio/PROTAC-Splitter-App).

## Table of Contents 📜

- [Installation](#installation)
- [Usage](#usage)
- [Data Availability](#data-availability)
- [Contributing](#contributing)
- [License](#license)

## Installation 🛠️

The package was tested in Python 3.10.8. Always use a virtual environment to install the package.

For using the code under the [scripts](scripts) directory in this repository, run the following commands:

```bash
git clone https://github.com/ribesstefano/PROTAC-Splitter.git
cd PROTAC-Splitter
pip install -r requirements.txt
pip install -r scripts/requirements.txt

# Add the package to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`/protac_splitter
```

Alternatively, you can install the package using pip (again, in a virtual environment):

```bash
pip install git+https://github.com/ribesstefano/PROTAC-Splitter.git
```

## Usage 🚀

To use the package, please refer to the function `split_protac` in the [protac_splitter/protac_splitter](protac_splitter/protac_splitter) module.

Here is an example of how to use the function:

```python
import pandas as pd
from protac_splitter import split_protac

# Split a PROTAC molecule reported as a SMILES string
protac_smiles = "CC(C)(C)S(=O)(=O)c1cc2c(Nc3ccc4scnc4c3)ccnc2cc1OCCOCCOCCOCCOCC(=O)Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O"
ligands = split_protac(protac_smiles)
print(ligands)

# One can also feed a DataFrame to the function
df = pd.read_csv("my/local/file.csv")
split_df = split_protac(df, protac_smiles_col="PROTAC SMILES")
print(split_df.head())
```

Alternatively, you can use the Gradio app at [scripts/protac_splitter_app.py](scripts/protac_splitter_app.py) have a GUI to split PROTAC molecules.

```sh
gradio scripts/protac_splitter_app.py
```

## Data Availability 📥

Curated public data, the synthetic PROTACs dataset, and trained models are available
for download from Zenodo at: [https://doi.org/10.5281/zenodo.15797309](https://doi.org/10.5281/zenodo.15797309).

## Contributing 🤝

We welcome contributions to this project! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
