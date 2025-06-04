import torch
import torch.nn as nn
import json
from torchvision.models import resnet18
from baku.model import BAKUTransformer


class TactileNonTNN(nn.Module):
    """
    Definition of the computation graph, which may or may not include temporal unrolling.
    """

    def __init__(self, cfg):
        super(TactileNonTNN, self).__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train
        self.test_cfg = cfg.test

        self.n_times = self.train_cfg.n_times
        self.non_tnn_model = self.train_cfg.non_tnn_model  # a string, to indicate the model name

        assert self.non_tnn_model is not None
        # assert self.n_times is None

        if self.non_tnn_model == 'resnet18':
            self.model = resnet18()
            input_channels = self.data_cfg.input_shape[0]
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.data_cfg.num_classes)
            print('calling resnet18 from pytorch')
            print(self.model)
        elif self.non_tnn_model == 'baku':
            self.model = BAKUTransformer(cfg=cfg)
        else:
            raise NotImplementedError

    def forward(self, x, cuda=False):
        return self.model(x)
        # if self.non_tnn_model == 'resnet18':
        #     return self.model(x)
        # elif self.non_tnn_model == 'baku':
        #     return self.model(x)
        # else:
        #     raise NotImplementedError
