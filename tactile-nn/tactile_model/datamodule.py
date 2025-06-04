from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import torch
from tactile_dataset import ShapeNetWhiskingDataset
from utils import data_transform


class FakeDataset(Dataset):  # for quick testing purpose
    def __init__(self, cfg, data_dir, train, transform):
        super(FakeDataset, self).__init__()
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        self.input_shape = (cfg.train.n_times,) + tuple(cfg.data.input_shape)
        self.num_classes = cfg.data.num_classes

    def __len__(self):
        return 10000 if self.train else 5000

    def __getitem__(self, idx):
        return torch.rand(self.input_shape), torch.randint(0, self.num_classes, (1,)).item()


class TactileDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.train_cfg = cfg.train

        self.data_dir = self.data_cfg.data_dir
        self.batch_size = self.train_cfg.batch_size
        self.num_workers = self.train_cfg.num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Download the dataset (if needed)
        pass

    def setup(self, stage=None):
        # data transformation
        transform = self.data_cfg.get("data_transform", None)
        assert transform is not None
        assert transform.name == 'ShapeNetReshape'
        assert isinstance(self.data_cfg.n_times, int)
        shapenet_transform = getattr(data_transform, transform.name)(**transform.kwargs)

        # Split the dataset into train/val datasets
        if stage == 'fit' or stage is None:
            # self.train_dataset = FakeDataset(cfg=self.cfg, data_dir=self.data_dir, train=True, transform=None)
            # self.val_dataset = FakeDataset(cfg=self.cfg, data_dir=self.data_dir, train=False, transform=None)

            self.train_dataset = ShapeNetWhiskingDataset(cfg=self.cfg, split="train", data_dir=self.data_dir,
                                                         data_transform=shapenet_transform)
            self.val_dataset = ShapeNetWhiskingDataset(cfg=self.cfg, split="validate", data_dir=self.data_dir,
                                                       data_transform=shapenet_transform)

        if stage == 'test' or stage is None:
            # self.test_dataset = FakeDataset(cfg=self.cfg, data_dir=self.data_dir, train=False, transform=None)

            self.test_dataset = ShapeNetWhiskingDataset(cfg=self.cfg, split="test", data_dir=self.data_dir,
                                                        data_transform=shapenet_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=4*self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=4*self.batch_size,
                          # num_workers=self.num_workers,
                          # pin_memory=True,
                          )
