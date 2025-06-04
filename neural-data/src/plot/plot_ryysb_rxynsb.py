import argparse
import numpy as np
import matplotlib.pyplot as plt

from dataset.rodgers_data import load_linregress_results
from utils.linregress_utils import process_rxynsb_ryysb_per_neuron, process_rxynsb_ryysb_pairwise_per_neuron

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--linregress_results", "-f", type=str, help="the linregress results .npz file")
    # parser.add_argument("--filter_ryy", action=argparse.BooleanOptionalAction, default=True, help="filter for r_yy > 0; use --no-filter_ryy to disable")
    parser.add_argument("--aggregate", type=str, default="argmax", help="how to join pairwise, max or mean]")
    args = parser.parse_args()

    # NOTE: size of test_r_xy_n_sb is (num_split_halves, num_train_test_splits, num_neurons)
    print("loading file...")
    results_dict = load_linregress_results(args.linregress_results)
    print("loaded file")

    filename = args.linregress_results.split("/")[-1].split(".")[0]

    if "pairwise" in filename:
        processed_rxynsb, processed_ryysb, target_animal_names = process_rxynsb_ryysb_pairwise_per_neuron(results_dict, filter_ryy=False, filter_rxx=False, aggregate=args.aggregate)
    else:
        processed_rxynsb, processed_ryysb, target_animal_names = process_rxynsb_ryysb_per_neuron(results_dict, filter_ryy=False, filter_rxx=False)


    plt.figure(figsize=(6, 8))
    plt.title(f"r_yy_sb vs r_xy_n_sb for {filename}")
    plt.xlim(0, 1)
    plt.xlabel("r_yy_sb")
    plt.ylabel("r_xy_n_sb")

    for animal_name, rxy, ryy in zip(target_animal_names, processed_rxynsb, processed_ryysb):
        print(animal_name, "rxy", rxy.shape, "ryy", ryy.shape)
        plt.scatter(ryy, rxy, label=animal_name)

    median_rxy = np.nanmedian(np.concatenate(processed_rxynsb))
    median_ryy = np.nanmedian(np.concatenate(processed_ryysb))
    print(f"median rxy: {median_rxy}")
    print(f"median ryy: {median_ryy}")
    plt.scatter(median_ryy, median_rxy, color='blue', marker='x', s=100,
                label=f"Median ({np.round(median_rxy, decimals=3)}, {np.round(median_ryy, decimals=3)})")

    plt.legend()

    out = f"./out/ryysb_vs_rxynsb_{filename}.png"
    plt.savefig(out, dpi=400)
    print("saved to", out)