# PROTAC Splitter

## Quickstart

Install the required dependencies and activate the relative environment:

```bash
conda env create -f environment.yml
conda activate env-protac-splitter
```

Unzip the `final.zip` file located under the `data` directory.

Run the following command for starting training the model:

```bash
mkdir -p models
python main.py \
    --output_dir="./models" \
    --data_dir="./data/final/" \
    --max_steps=-1 \
    --num_train_epochs=50
```

One can pass additional arguments to push to an Hugging Face repository once training completes:

```bash
python main.py \
    --output_dir="./models" \
    --data_dir="./data/final/" \
    --max_steps=-1 \
    --num_train_epochs=50 \
    --hub_token="my-unforgettable-token" \
    --organization="my-awesome-organization" \
```

In general, refer to the help message for more information about the command line arguments:

```bash
python main.py --help
```