import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm 

from dataset.rodgers_data import load_linregress_results
from utils.linregress_utils import process_rxynsb_ryysb_per_neuron, process_rxynsb_ryysb_pairwise_per_neuron
from concurrent.futures import ProcessPoolExecutor


def process_file(file, args):
    try:
        data_dict = load_linregress_results(file)
    except Exception as error:
        print("could not open file", file)
        print(error)
        return None
    if "pairwise" in file:
        processed_rxynsb, _, _ = process_rxynsb_ryysb_pairwise_per_neuron(data_dict, args.filter_ryy, args.aggregate)
    else:
        processed_rxynsb, _, _ = process_rxynsb_ryysb_per_neuron(data_dict, args.filter_ryy)
    processed_rxynsb = np.concatenate(processed_rxynsb)

    name = file.split("results_")[-1].split(".")[0]
    return name, {
        "median": np.nanmedian(processed_rxynsb),
        "sem": np.nanstd(processed_rxynsb) / np.sqrt(np.sum(~np.isnan(processed_rxynsb)))
    }


def get_median_sem_from_results(results_dir, cpus=1, args={}):
    if cpus == 1:
        files = glob(results_dir + "/*.npz")
        output = [process_file(file, args) for file in files]
        results = {name: result for (name, result) in output}
    else:
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            files = glob(results_dir + "/*.npz")
            output = [item for item in executor.map(process_file, files, [args]*len(files)) if item]
            results = {name: result for (name, result) in output}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare linear regression results.")
    parser.add_argument("--out", type=str, default="./out/linregress_results_comparison.png", help="Path to save the results npz file.")
    parser.add_argument("--results_dir", "-d", type=str, default="/data/group_data/neuroagents_lab/neural_consistency/somatosensory", help="Filepath to load the data from")
    parser.add_argument("--json", type=str, help="json filepath from previous run to load the data from")
    parser.add_argument("--cpus", type=int, default=1, help="Number of CPUs to use for parallel processing.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[15, 8], help="plot figure size")
    parser.add_argument("--sortby", type=str, default="name", help="sort by name or median")
    parser.add_argument("--filter_ryy", type=bool, default=True, help="filter out r_yy < 0")
    parser.add_argument("--aggregate", type=str, default="max", help="how to join pairwise, max or mean]")
    args = parser.parse_args()
    
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            results = json.load(f)
        for k,v in results.items():
            if not results[k]['median']:
                results[k]['median'] = np.nan
                results[k]['sem'] = np.nan
    else:
        results = get_median_sem_from_results(args.results_dir, args.cpus, args)

        with open(args.out.replace(".png", ".json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    if args.sortby == "name":
        def sort_key(item):
            val = item
            if "ridge" in item:
                val = float(item.split("_")[-1])
                if "pairwise" in item:
                    val += 1e12
            return val
        names = sorted(results.keys(), key=lambda item: float(item.split("_")[-1]) if "ridge" in item else item)
    elif args.sortby == "median":
        names = sorted(results.keys(), key=lambda k: float(results[k]["median"]))
    else:
        names = results.keys()

    medians = [results[name]["median"] for name in names]
    sems = [results[name]["sem"] for name in names]

    plt.figure(figsize=args.figsize)
    plt.errorbar(names, medians, yerr=sems, fmt='o', capsize=3)
    plt.xticks(rotation=90)
    plt.xlabel('Linregress Type')
    plt.ylabel('Median')
    plt.title('Median and SEM of Linregress Results')
    plt.tight_layout()

    plt.savefig(args.out, dpi=300)
    print("figure saved to", args.out)


