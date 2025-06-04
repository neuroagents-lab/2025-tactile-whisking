from itertools import combinations
import numpy as np

def make_splits(train_frac, num_stimuli=6, idx=None):
    if not hasattr(idx, '__iter__'):
        idx = range(num_stimuli)
    splits = []
    for split in combinations(idx, int(train_frac * num_stimuli)):
        train_idx = np.array(split)
        test_idx = np.array([i for i in range(num_stimuli) if i not in train_idx])
        splits.append({"train": train_idx, "test": test_idx})
    return splits

def process_rxynsb_ryysb_pairwise_per_neuron(data_dict, filter_ryy=True, filter_rxx=True, aggregate="argmax"):
    target_animal_names = []
    processed_rxysb = []
    processed_ryysb = []

    per_animal = {}
    for pair_name, animal_data in data_dict.items():
        source_animal, target_animal = pair_name.split('->')
        target_animal_names.append(target_animal)
        rxysb, ryysb = get_rxy_ryy_filtered(animal_data["test"], filter_ryy, filter_rxx)
        
        if target_animal not in per_animal:
            per_animal[target_animal] = {
                "rxysb": [],
                "ryysb": []
            }
        per_animal[target_animal]["rxysb"].append(np.nanmean(rxysb, axis=(0, 1)))
        per_animal[target_animal]["ryysb"].append(np.nanmean(ryysb, axis=(0, 1)))

    for pair_data in per_animal.values():
        if aggregate == "argmax":
            median_rxysb_per_pair = [np.nanmedian(pair_rxysb) for pair_rxysb in pair_data["rxysb"]]
            best_pair_animal_idx = np.argmax(median_rxysb_per_pair)
            processed_rxysb.append(pair_data["rxysb"][best_pair_animal_idx])
            processed_ryysb.append(pair_data["ryysb"][best_pair_animal_idx])
        elif aggregate == "mean":
            processed_rxysb.append(np.array([np.mean(xy) for xy in pair_data["rxysb"]]))
            processed_ryysb.append(np.array([np.mean(yy) for yy in pair_data["ryysb"]]))
        elif aggregate == "concat":
            processed_rxysb.append(np.concatenate(pair_data["rxysb"]))
            processed_ryysb.append(np.concatenate(pair_data["ryysb"]))
        else:
            raise NotImplementedError(f"aggregate={aggregate} is not valid")

    return processed_rxysb, processed_ryysb, target_animal_names


def process_rxynsb_ryysb_per_neuron(data_dict, filter_ryy=True, filter_rxx=True):
    target_animal_names = []
    processed_rxysb = []
    processed_ryysb = []

    for animal_name, animal_data in data_dict.items():
        target_animal_names.append(animal_name)
        rxysb, ryysb = get_rxy_ryy_filtered(animal_data["test"], filter_ryy, filter_rxx)
        processed_rxysb.append(np.nanmean(rxysb, axis=(0, 1)))
        processed_ryysb.append(np.nanmean(ryysb, axis=(0, 1)))

    return processed_rxysb, processed_ryysb, target_animal_names


def get_rxy_ryy_filtered(test_data, filter_ryy, filter_rxx, cutoff=0.5):
    if not (filter_ryy or filter_rxx):
        return test_data["r_xy_n_sb"], test_data["r_yy_sb"]

    mask = np.zeros_like(test_data["r_yy"], dtype=bool)
    if filter_ryy:
        mask |= test_data["r_yy"] < cutoff
    if filter_rxx:
        mask |= test_data["r_xx"] < cutoff

    return (np.where(mask, np.nan, test_data["r_xy_n_sb"]), 
            np.where(mask, np.nan, test_data["r_yy_sb"]))
