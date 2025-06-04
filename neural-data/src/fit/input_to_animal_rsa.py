import os
import argparse

import numpy as np
from tqdm import tqdm

from brainmodel_utils.metrics.consistency import get_linregress_consistency
from utils.linregress_utils import make_splits
from dataset.rodgers_data import load_rodgers_data, concatenate_sessions

from itertools import permutations
from animal_to_animal_rsa import get_rsa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear regression analysis on Rodgers data.")
    parser.add_argument("--out", type=str, default="./out/model_input_rsa.npz", help="Path to save the results npz fie.")
    parser.add_argument("--data", type=str, default="/data/group_data/neuroagents_lab/tactile/neural_data/rodgers6_simulated.npz", help="Filepath to load the data from")
    parser.add_argument("--animal_data", type=str, default="./data/rodgers6_data.npz", help="Filepath to load the data from")
    parser.add_argument("--cpus", type=int, default=1, help="number of parallel jobs to run")
    parser.add_argument("--metric", type=str, choices=["rsa_pearsonr", "rsa_spearmanr"], default="rsa_pearsonr", help="rsa_pearsonr or rsa_spearmanr")
    args = parser.parse_args()

    input_data = np.load(args.data)['data']
    input_data = input_data.mean(axis=1)  # shape: (6, 30, 5, 7)
    input_data = input_data.reshape(6, -1)  # shape: (6, 30*5*7)
    print("input_data", input_data.shape)

    data_per_animal = load_rodgers_data(path=args.animal_data)
    animals = concatenate_sessions(list(data_per_animal.values()))
    print("animals", animals.shape)

    result = get_rsa(input_data, animals, args.cpus, args.metric)
    print("result", result)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, result)
    print("Saved results to", args.out)