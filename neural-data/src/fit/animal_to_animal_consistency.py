import os
import argparse

import numpy as np
from tqdm import tqdm

from brainmodel_utils.metrics.consistency import get_linregress_consistency
from utils.linregress_utils import make_splits
from dataset.rodgers_data import load_rodgers_data, concat_other_animals

from itertools import permutations


NUM_STIMULI = 6

def process_animal(source, target, args):
    splits = make_splits(0.5, NUM_STIMULI)

    linregress_args = dict(
        source=source,
        target=target,
        num_parallel_jobs=args.cpus,
        metric="pearsonr",
        splits=splits,
        num_bootstrap_iters=1000
    )
    map_kwargs = {}
    if args.map_type == "percentile":
        map_kwargs["map_type"] = "percentile"

    elif args.map_type == "elasticnet":
        map_kwargs["map_type"] = "sklinear"
        map_kwargs["map_kwargs"] = {
            "regression_type": "ElasticNet",
            "regression_kwargs": {
                "alpha": args.map_arg[0],
                "l1_ratio": args.map_arg[1]
            },
        }
    elif args.map_type == "ridge":
        map_kwargs["map_type"] = "sklinear"
        map_kwargs["map_kwargs"] = {
            "regression_type": "Ridge",
            "regression_kwargs": {"alpha": args.map_arg[0]},
        }
    elif args.map_type == "pls":
        map_kwargs["map_type"] = "pls"
        map_kwargs["map_kwargs"] = {
            "n_components": int(args.map_arg[0]),
        }
    else:
        raise ValueError(f"Unsupported map_type: {args.map_type}")
    
    linregress_args["map_kwargs"] = map_kwargs

    return get_linregress_consistency(**linregress_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear regression analysis on Rodgers data.")
    parser.add_argument("--out", type=str, default="./out/linregress_results.npz", help="Path to save the results npz fie.")
    parser.add_argument("--data", type=str, default="./data/rodgers6_data.npz", help="Filepath to load the data from")
    parser.add_argument("--map_type", type=str, default="percentile", help="map_type in map_kwargs")
    parser.add_argument("--map_arg", type=float, nargs='+', default=[0], help="arg for map_kwargs, depending on map_type")
    parser.add_argument("--cpus", type=int, default=1, help="number of parallel jobs to run")
    parser.add_argument("--pairwise", action="store_true", help="run pairwise instead of many to one")
    args = parser.parse_args()

    data_per_animal = load_rodgers_data(path=args.data) # load from preprocessed

    # NOTE: we could run process_animal in parallel, but every get_linregress_consistency
    #       runs in parallel anyway so there is no speedup
    # ------
    # num_animals = len(data_per_animal)
    # n_jobs = min(num_animals, args.cpus)
    # cpu_per_job = max(1, args.cpus // n_jobs)

    # results = Parallel(n_jobs=n_jobs, return_as="generator")(
    #     delayed(process_animal)(animal_id, data_per_animal, cpu_per_job, args.mode)
    #     for animal_id in data_per_animal.keys()
    # )
    # results_dict = {animal_id: result for animal_id, result in list(tqdm(results, total=num_animals))}
    # ------

    # results_dict = {}
    # for animal_id in tqdm(data_per_animal.keys()):
    #     result = process_animal(animal_id, data_per_animal, args)
    #     results_dict[animal_id] = result
    results_dict = {}
    if args.pairwise:
        pairs = permutations(data_per_animal.keys(), 2)
        for animal1, animal2 in tqdm(pairs):
            result = process_animal(data_per_animal[animal1], data_per_animal[animal2], args)
            results_dict[animal1 + "->" + animal2] = result
    else:
        for animal_id in tqdm(data_per_animal.keys()):
            rest_of_animals = concat_other_animals(animal_id, data_per_animal)
            result = process_animal(rest_of_animals, data_per_animal[animal_id], args)
            results_dict[animal_id] = result

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **results_dict)
    print("Saved results to", args.out)