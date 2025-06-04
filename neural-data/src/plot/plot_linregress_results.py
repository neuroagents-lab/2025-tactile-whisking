import argparse
import matplotlib.pyplot as plt
import numpy as np

from dataset.rodgers_data import load_linregress_results
from utils.plotting import plot_linregress_result


def process_r_sb(r_sb):
    # print(r_sb.shape)
    r_sb = np.mean(r_sb, axis=0)
    # print("nans:", np.isnan(r_sb).sum(), "out of", r_sb.size)
    r_sb = r_sb[~np.isnan(r_sb)]
    return r_sb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear regression analysis on Rodgers data.")
    parser.add_argument("--resultfile", type=str, help="Linregress results .npz filepath")
    parser.add_argument("--out", type=str, default="./out/6stimuli", help="Path to save the results and plots.")
    parser.add_argument("--all", action="store_true", help="Use flag to plot each pair instead of only grouped over all pairs.")
    args = parser.parse_args()

    all_results = load_linregress_results(args.resultfile, allow_pickle=True)
    grouped = {"train": {"r_xx": [], "r_yy": [], "r_xy": [], "r_xy_n_sb": []},
                "test": {"r_xx": [], "r_yy": [], "r_xy": [], "r_xy_n_sb": []}}
    for title, results in all_results.items():
        results = results.item()
        if args.all:
            plot_linregress_result(results, title, path=args.out)
        for split, split_result in results.items():
            for key, result in split_result.items():
                if key in grouped[split]:
                    print(key, split, result.shape)
                    avg_over_trials = np.mean(result, axis=0)
                    avg_over_stimuli = np.mean(avg_over_trials, axis=0)
                    grouped[split][key].append(avg_over_stimuli)
                    
                    
    num_pairs = len(grouped["train"]["r_xx"])
    
    for split in grouped:
        for key in grouped[split]:
            grouped[split][key] = np.concatenate(grouped[split][key])

    plot_linregress_result(grouped, f"over {num_pairs} pairs", path=args.out)
