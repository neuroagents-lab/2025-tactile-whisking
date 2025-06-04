import os
import argparse

import numpy as np
from tqdm import tqdm

from brainmodel_utils.metrics.consistency import get_linregress_consistency
from utils.linregress_utils import make_splits
from dataset.rodgers_data import load_rodgers_data, concat_other_animals

from itertools import permutations

def get_rsa(source, target, cpus, metric="rsa_pearsonr"):
    return get_linregress_consistency(
        source=source,
        target=target,
        num_parallel_jobs=cpus,
        metric=metric,
        map_kwargs={"map_type": "identity"},
        splits=make_splits(train_frac=0.0, num_stimuli=6),
        num_bootstrap_iters=1000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear regression analysis on Rodgers data.")
    parser.add_argument("--out", type=str, default="./out/linregress_results.npz", help="Path to save the results npz fie.")
    parser.add_argument("--data", type=str, default="./data/rodgers6_data.npz", help="Filepath to load the data from")
    parser.add_argument("--cpus", type=int, default=1, help="number of parallel jobs to run")
    parser.add_argument("--pairwise", action="store_true", help="run pairwise instead of many to one")
    parser.add_argument("--metric", type=str, choices=["rsa_pearsonr", "rsa_spearmanr"], help="rsa_pearsonr or rsa_spearmanr")
    args = parser.parse_args()

    data_per_animal = load_rodgers_data(path=args.data) # load from preprocessed

    results_dict = {}
    if args.pairwise:
        pairs = permutations(data_per_animal.keys(), 2)
        for animal1, animal2 in tqdm(pairs):
            result = get_rsa(data_per_animal[animal1], data_per_animal[animal2], args.cpus, args.metric)
            results_dict[animal1 + "->" + animal2] = result
    else:
        for animal_id in tqdm(data_per_animal.keys()):
            rest_of_animals = concat_other_animals(animal_id, data_per_animal)
            result = get_rsa(rest_of_animals, data_per_animal[animal_id], args.cpus, args.metric)
            results_dict[animal_id] = result

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **results_dict)
    print("Saved results to", args.out)