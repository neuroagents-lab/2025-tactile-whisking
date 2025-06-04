import os
import argparse

import numpy as np

from utils.linregress_utils import make_splits, get_rxy_ryy_filtered
from brainmodel_utils.metrics.consistency import get_linregress_consistency
from dataset.rodgers_data import load_rodgers_data, concatenate_sessions
from utils.model_eval import load_activation_data


def neural_eval(model_data, animal_neurons, metric, cpus=1):
    assert model_data.ndim == 2, "expecting input (stimuli, units)"

    num_stimuli = model_data.shape[0]

    if "rsa" in metric:
        linregress_args = dict(
            metric=metric,
            splits=make_splits(0.0, num_stimuli=num_stimuli),
            map_kwargs={"map_type": "identity"},
        )
    else:
        assert metric in ["pearsonr", "spearmanr"]
        linregress_args = dict(
            metric=metric,
            splits=make_splits(0.5, num_stimuli=num_stimuli),
            map_kwargs={
            "map_type": "sklinear",
                "map_kwargs": {
                    "regression_type": "Ridge",
                    "regression_kwargs": { "alpha": 5e7 }
            }}
        )
    return get_linregress_consistency(
        source=model_data,
        target=animal_neurons,
        num_parallel_jobs=cpus,
        num_bootstrap_iters=1000,
        **linregress_args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear regression analysis on Rodgers data.")
    parser.add_argument("--outdir", "-o", type=str, default="./out", help="Path to save the results npz files.")
    parser.add_argument("--data", type=str, default="./data/rodgers6_data.npz", help="Filepath to load the data from")
    parser.add_argument("--input", "-i", type=str, default="./data/activations/tactile1000hz_enc_Zhuang_att_Mamba_sgd_steplr_lr1e_3/activations.npz", help="Filepath to load the model activation data from")
    parser.add_argument("--metric", type=str, default="pearsonr", help="metric function")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="skip existing files")
    parser.add_argument("--cpus", type=int, default=1, help="number of parallel jobs to run")
    args = parser.parse_args()

    data_per_animal = load_rodgers_data(path=args.data) # load from preprocessed
    data_per_animal['concatenated'] = concatenate_sessions(list(data_per_animal.values()))

    print("loading", args.input)
    layer_activations = load_activation_data(args.input)
    print("layers:", ", ".join(layer_activations.keys()))

    for layer_name, layer_data in layer_activations.items():
        print("\n-----", layer_name, layer_data.shape, "-----")
        for animal_name, animal_data in data_per_animal.items():
            print()
            print(animal_name, animal_data.shape)

            input_dirname = os.path.basename(os.path.dirname(args.input))
            outfile = os.path.join(args.outdir, input_dirname, layer_name, animal_name + ".npz")
            if args.skip_existing and os.path.exists(outfile):
                print("skipping", outfile)
                continue

            results = neural_eval(layer_data, animal_data, metric=args.metric, cpus=args.cpus)

            rxysb, ryysb = get_rxy_ryy_filtered(results["test"], filter_ryy=True, filter_rxx=False)
            median_rxy = np.nanmedian(np.nanmean(rxysb, axis=(0, 1)))
            median_ryy = np.nanmedian(np.nanmean(ryysb, axis=(0, 1)))
            print(f"median rxy: {median_rxy}")
            print(f"median ryy: {median_ryy}")

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            np.savez(outfile, **results)
            print("Saved results to", outfile)

                    
