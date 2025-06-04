import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import uuid
from tqdm import tqdm
import re
import json


def get_sweep_info(dirname, data_version="1000hz"):
    if data_version == "110hz":
        matches = re.findall(r'(h(-?\d+(.\d)?))|(to\d+)|(t\d+)|(v\d+)|(s\d+)', dirname.replace('_', '.'))
        if matches and len(matches) == 5:
            h = float(matches[0][0][1:])
            to = int(matches[1][3][2:])
            t = int(matches[2][4][1:])
            v = int(matches[3][5][1:])
            s = int(matches[4][6][1:])
            return {"h":h, "to":to, "t":t, "v":v, "s":s}
    else:
        pattern = r"h(-?\d+)_d(-?\d+)_t(-?\d+)"
        match = re.search(pattern, dirname)
        if match:
            h = match.group(1)
            d = match.group(2)
            t = match.group(3)
            return {"h": h, "d": d, "t": t}
    return {}




def pad_whiskers_to_35(whisker_tensor):
    """ Given (110, 30, 6) will reshape to (110, 35, 6) """
    positions_to_insert_zeros = [5, 6, 12, 13, 34]
    padded_whisker = np.zeros((110, 35, 6))
    all_positions = list(range(35))
    non_zero_positions = [
        pos for pos in all_positions if pos not in positions_to_insert_zeros
    ]
    padded_whisker[:, non_zero_positions, :] = whisker_tensor
    return padded_whisker


def forward_fill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(arr.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

def ffill_tensor(tensor):
    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[2]):
            tensor[:, i, j] = forward_fill(tensor[:, i, j])
    return tensor



def preprocess_tactile_data(
    base_dir, output_dir, categories_dict, n_cpus=64, process_values=False, data_version='1000hz'
):
    print("Preprocessing", base_dir, "to", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = [
        cat
        for cat in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, cat))
    ]
    tasks = []

    for category_id in categories:
        category_dir = os.path.join(base_dir, category_id)

        sub_ids = [
            sub_id
            for sub_id in os.listdir(category_dir)
            if os.path.isdir(os.path.join(category_dir, sub_id))
        ]
        for sub_id in sub_ids:
            sub_id_dir = os.path.join(category_dir, sub_id)

            processed_file = os.path.join(
                output_dir, f"{category_dir}_{sub_id}_processed.txt"
            )
            if not os.path.exists(processed_file):
                tasks.append(
                    (
                        sub_id_dir,
                        output_dir,
                        sub_id,
                        category_dir,
                        categories_dict,
                        process_values,
                        data_version
                    )
                )
            else:
                print(f"skipping {category_dir}/{sub_id} (already processed)")

    if n_cpus == 1:
        print('Running non parallel')
        for task in tqdm(tasks):
            process_task(task)
    else:
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            executor.map(process_task, tasks)


def process_task(args):
    (
        sub_id_dir,
        output_split_dir,
        sub_id,
        category_dir,
        categories_dict,
        process_values,
        data_version
    ) = args

    dynamics_dirs = glob(os.path.join(sub_id_dir, "*", "dynamics"))

    for dynamics_dir in dynamics_dirs:
        sweep_data = []
        for force_file in ["Fx.csv", "Fy.csv", "Fz.csv", "Mx.csv", "My.csv", "Mz.csv"]:
            file_path = os.path.join(dynamics_dir, force_file)
            data = pd.read_csv(file_path, header=None).values
            sweep_data.append(data)

        sweep_tensor = np.stack(sweep_data, axis=-1)
        assert sweep_tensor.shape[1] == 30 and sweep_tensor.shape[2] == 6

        if sweep_tensor.shape[0] != 110:
            step_size = sweep_tensor.shape[0] // 110
            sweep_tensor = sweep_tensor[1::step_size][:110]

        if process_values:
            sweep_tensor = np.nan_to_num(sweep_tensor, nan=0)
            sweep_tensor[np.abs(sweep_tensor) > 1000] = np.nan
            sweep_tensor = ffill_tensor(sweep_tensor)
            sweep_tensor = np.nan_to_num(sweep_tensor, nan=0)
            sweep_tensor = pad_whiskers_to_35(sweep_tensor)

        sweepname = dynamics_dir.split("/")[-2]
        short_uid = str(uuid.uuid4())[:8]
        shape_id = category_dir.split("/")[-1] + "/" + sub_id
        new_filename = f"{shape_id.replace('/', '_')}_{sweepname}_{short_uid}.npz"
        new_filepath = os.path.join(output_split_dir, new_filename)
        metadata = get_sweep_info(sweepname, data_version=data_version)
        label = categories_dict[shape_id] if shape_id in categories_dict else "?"
        np.savez_compressed(
            new_filepath,
            data=sweep_tensor,
            shape=shape_id,
            label=int(label),
            **metadata
        )

        processed_file = os.path.join(
            output_split_dir, f"{category_dir}_{sub_id}_processed.txt"
        )
        with open(processed_file, "w") as f:
            pass


def rename_files_sequentially(output_dir):
    npz_files = glob(os.path.join(output_dir, "*.npz"))
    npz_files.sort()

    for idx, file_path in enumerate(npz_files):
        new_filename = f"{idx:08d}.npz"
        new_filepath = os.path.join(output_dir, new_filename)
        os.rename(file_path, new_filepath)
        print(f"Renaming {file_path} to {new_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess tactile whisker data.")
    parser.add_argument(
        "--categories_file",
        type=str,
        default="/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/categories.json",
        help="Path to the categories JSON file.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/raw/train",
        help="Base directory containing raw tactile data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/processed/train",
        help="Directory to save processed data.",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=64,
        help="Number of CPUs to use for parallel processing.",
    )
    parser.add_argument(
        "--process_values",
        action="store_true",
        help="Flag to enable processing values (e.g., enforcing first timestep to zeros).",
    )
    parser.add_argument(
        "--data_version",
        type=str,
        default="1000hz",
        help="data version to figure out how to parse it"
    )

    args = parser.parse_args()

    with open(args.categories_file, "r") as file:
        categories_dict = json.load(file)

    preprocess_tactile_data(
        args.base_dir,
        args.output_dir,
        categories_dict,
        args.n_cpus,
        args.process_values,
        args.data_version
    )

    rename_files_sequentially(args.output_dir)
