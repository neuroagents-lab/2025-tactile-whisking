import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from glob import glob

from dataset.rodgers_data import load_linregress_results
from utils.linregress_utils import get_rxy_ryy_filtered

def process_file(filepath):
    try:
        results = load_linregress_results(filepath)
    except Exception as error:
        print("could not open file", filepath)
        print(error)
        return None

    rxysb, ryysb = get_rxy_ryy_filtered(results["test"], filter_ryy=True, filter_rxx=True, cutoff=0.0)
    mean_rxysb = np.nanmean(rxysb, axis=(0, 1))
    mean_ryysb = np.nanmean(ryysb, axis=(0, 1))

    rxysb_filtered, ryysb_filtered = get_rxy_ryy_filtered(results["test"], filter_ryy=True, filter_rxx=True, cutoff=0.5)
    filtered_mean = np.nanmean(rxysb_filtered, axis=(0, 1))
    median_rxy = np.nanmedian(filtered_mean) if np.sum(~np.isnan(filtered_mean)) > 0 else np.nan

    #                       3 for model/layer/target     .npz
    filename = ("-".join(filepath.split("/")[-3:])).split(".")[0]
    shortname = re.sub(r"(tactile1000hz_)|(_lr1e_.)|(_rot_tflip110)", "", filename)
    return filename, { 
        "rxy": mean_rxysb,
        "ryy": mean_ryysb,
        "median": median_rxy
        }

def get_neurons_from_results(args={}):
    results_dir = args.results_dir
    cpus = args.cpus

    if cpus == 1:
        files = glob(results_dir + "/*/*/*.npz")
        output = [process_file(file) for file in files]
        results = {name: result for (name, result) in output}
    else:
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            files = glob(results_dir + "/*/*/*.npz")
            output = [item for item in executor.map(process_file, files) if item]
            results = {name: result for (name, result) in output}

    return results


def make_results_dict(results):
    results_dict = {}
    for name, result in results.items():
        name_split = name.split("-")
        model_name = name_split[0]
        layer_name = name_split[1]

        if model_name not in results_dict:
            results_dict[model_name] = {}
        if layer_name not in results_dict[model_name]:
            results_dict[model_name][layer_name] = []
        results_dict[model_name][layer_name].append(result)
    
    return results_dict


def get_max_per_model(results):
    results_dict = make_results_dict(results)

    model_results = {}
    for model_name, layers in results_dict.items():
        model_results[model_name] = max(
            (res for layer in layers.values() for res in layer),
            key=lambda x: x["median"]
        )
    
    return model_results

def get_max_per_layer_per_model(results):
    results_dict = make_results_dict(results)
    all_layer_results = {}
    for model_name, layers in results_dict.items():
        layer_results = {}
        for layer_name, layer_data in layers.items():
            layer_results[layer_name] = max(
                [res for res in layer_data],
                key=lambda x: x["median"]
            )
        all_layer_results[model_name] = layer_results
    return all_layer_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", "-d", type=str, default="/data/group_data/neuroagents_lab/tactile/model_to_animal/linregress/5e7", help="directory to load the data from")
    parser.add_argument("--filter_ryy", action=argparse.BooleanOptionalAction, default=True, help="filter for r_yy > 0; use --no-filter_ryy to disable")
    parser.add_argument("--cpus", type=int, default=1, help="Number of CPUs to use for parallel processing.")
    parser.add_argument("--out", type=str, default="./out/ryysb_vs_rxynsb_models.png", help="Path to dir save the output image/json.")
    args = parser.parse_args()

    # NOTE: size of test_r_xy_n_sb is (num_split_halves, num_train_test_splits, num_neurons)
    results_dict = get_neurons_from_results(args)
    model_results = get_max_per_layer_per_model(results_dict)

    for name, model_result in model_results.items():
        name = name.replace("tactile100hz_", "")
        plt.figure(figsize=(8, 8))
        plt.title(f"r_yy_sb vs r_xy_n_sb for {name}")
        plt.xlim(0, 1)
        plt.xlabel("r_yy_sb")
        plt.ylabel("r_xy_n_sb")

        for layer_name, result in model_result.items():
            mean_rxysb = result["rxy"]
            mean_ryysb = result["ryy"]
            plt.scatter(mean_ryysb, mean_rxysb, label=name)

            median_rxy = np.nanmedian(mean_rxysb)
            median_ryy = np.nanmedian(mean_ryysb)
            print(f"median rxy: {median_rxy}")
            print(f"median ryy: {median_ryy}")
            plt.scatter(median_ryy, median_rxy, color='blue', marker='x', s=100,
                    label=f"Median ({np.round(median_rxy, decimals=3)}, {np.round(median_ryy, decimals=3)})")

        plt.legend()

        filename = args.out.replace(".png", f"_{name}.png")
        plt.savefig(filename, dpi=400)
        print("saved to", filename)