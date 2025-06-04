import os
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
from typing import Callable


class ShapeNetWhiskingDataset(Dataset):
    """
    Class for loading the tactile_whiskers dataset

    Arguments:
        data_dir        : (string) directory of tactile_whiskers dataset
        split           : (string) subdirectory (train/test/validate)
        include_metadata: (bool) set False for each item to return (data, label)
                                 set True to load (data, label, metadata)
    """

    def __init__(self, cfg, data_dir, data_transform: Callable, split="train", include_metadata=False):
        super(ShapeNetWhiskingDataset, self).__init__()
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Error: Tried to load dataset from {data_dir} which does not exist."
            )

        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train
        self.test_cfg = cfg.test

        self.include_metadata = include_metadata
        self.data_dir = f"{data_dir}/{split}"
        self.split = split

        self.sweep_files = glob(os.path.join(self.data_dir, "*.npz"))
        self.dataset_len = len(self.sweep_files)

        self.data_transform = data_transform

    def __getitem__(self, index):
        """
        Each item represents a sweep, a tensor containing (timesteps, whiskers, force xyz torque xyz) = (110, 35, 6)
        Returns (tensor, category_label)
        """
        try:
            data_file = np.load(self.sweep_files[index])
            data = torch.from_numpy(data_file["data"]).to(torch.float32)
            data = self.data_transform(data=data)
            label = int(data_file["label"].item())

            if self.include_metadata:
                metadata = {key: data_file[key] for key in data_file if key not in {'data', 'label'}}
                return data, label, index, metadata
            else:
                return data, label, index
        except Exception as error:
            raise type(error)(f"Encountered while loading file {self.sweep_files[index]}: {error}")

    def __len__(self):
        return self.dataset_len
