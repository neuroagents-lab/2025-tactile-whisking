# Tactile Neural Network Models

We use [PyTorchTNN (`pt-tnn`)](https://github.com/neuroagents-lab/PyTorchTNN) to construct our models.

Code for our Encoder-Attender-Decoder (EAD) architecture is under [tactile_model/enc_att_dec](./tactile_model/enc_att_dec/)

## Installation

This project uses the [poetry](https://python-poetry.org) package manager.

1. Activate your preferred virtual environment with Python3.12+
```
# conda
conda create -n tactilenn python=3.12
conda activate tactilenn

# poetry
poetry env use python3.12
```
2. `poetry install`


## Running code

Config management is handled using [hydra](https://hydra.cc/docs/intro/).

The model training code is in [tactile_model/main.py](./tactile_model/main.py)

See bash scripts (`*.sh`) for detailed examples on how to run.
