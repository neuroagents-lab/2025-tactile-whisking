# Neural Data & Evaluation

We evaluated models' neural fit score against the Rodgers 2022 mouse barrel cortex dataset
([paper](https://www.nature.com/articles/s41597-022-01728-1), [data](https://dandiarchive.org/dandiset/000231/0.220904.1554)).

## Installation

This project uses the [poetry](https://python-poetry.org) package manager.
1. Activate your preferred virtual environment with Python3.12+
```
# conda
conda create -n neural python=3.12
conda activate neural

# poetry
poetry env use python3.12
```
2. `poetry install`


## Running code

Run tests using `pytest`.

Some data has already been preprocessed and saved in [data](./data/).
Neural fitting is under [src/fit](./src/fit/).