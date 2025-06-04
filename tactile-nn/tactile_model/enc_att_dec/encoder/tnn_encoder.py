import torch
import torch.nn as nn
from pt_tnn.temporal_graph import TemporalGraph
from pt_tnn.pre_post_memory import FullyConnected
import json
from utils import input_transform
from omegaconf import OmegaConf

__all__ = ["TNNEncoder"]


class TNNEncoder(nn.Module):
    """
    Definition of the computation graph, including reading the graph from the
    configuration file and unrolling the graph in time.
    """

    def __init__(self, cfg):
        super(TNNEncoder, self).__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.encoder_cfg = cfg.model.encoder

        self.n_times = self.encoder_cfg.n_times
        data_n_times = self.data_cfg.get("n_times", None)
        if data_n_times is not None:
            assert self.n_times == data_n_times
            # if there is a time dimension in the data, make sure it matches unroll time-steps
        self.full_unroll = self.encoder_cfg.full_unroll

        self.return_all = self.encoder_cfg.return_all

        config = self.override_model_config()

        transform = self.encoder_cfg.get("input_transform", None)
        if transform is None or transform.name is None:
            self.transform = None
        else:
            self.transform = getattr(input_transform, transform.name)(**transform.kwargs)

        self.TG = TemporalGraph(model_config_file=config,
                                input_shape=self.data_cfg.input_shape, num_timesteps=self.n_times,
                                transform=self.transform)

    def override_model_config(self):
        # override the `input_shape` and `num_timesteps` from the model_config_file
        with open(self.encoder_cfg.model_config_file, "r") as f:
            config = json.load(f)

        output_node = config["output_node"]
        output_node_config = None
        for node_config in config["nodes"]:
            if node_config["name"] == output_node:
                output_node_config = node_config

        output_node_config["out_channels"] = self.model_cfg.repr_dim

        output_pre_mem, output_post_mem = output_node_config["pre_memory"], output_node_config["post_memory"]

        def modify_out_channels(mem, num_classes):
            modified = False
            if mem["name"] == "Sequential":
                last_module_config = mem["list_of_func_kwargs"][-1]
                assert last_module_config["name"] == "FullyConnected"
                last_module_config["out_channels"] = num_classes
                modified = True
            else:  # single module, can be MaxPool, FullyConnected, Conv2dCell, etc.
                if "out_channels" in mem:
                    mem["out_channels"] = num_classes
                    modified = True
            return modified

        # first try to find prediction head in post_mem
        modified = modify_out_channels(mem=output_post_mem, num_classes=self.model_cfg.repr_dim)
        if not modified:  # then try to find prediction head in pre_mem
            modified = modify_out_channels(mem=output_pre_mem, num_classes=self.model_cfg.repr_dim)

        if not modified:
            print('no prediction layers are found in the output node config, please make sure the model is correct')
            print('below is the modified version of the output node:')
            print(output_node_config)
            raise Exception

        return config

    def forward(self, x, return_activations=False):
        # x: (b, t, C, H, W) or (b, C, H, W)
        if self.full_unroll:
            if return_activations:
                output, activations = self.TG.full_unroll(x, n_times=self.n_times, return_all=self.return_all,
                                                          return_activations=True)
            else:
                output = self.TG.full_unroll(x, n_times=self.n_times, return_all=self.return_all,
                                             return_activations=False)
        else:
            if return_activations:
                output, activations = self.TG(x, n_times=self.n_times, return_all=self.return_all,
                                              return_activations=True)
            else:
                output = self.TG(x, n_times=self.n_times, return_all=self.return_all, return_activations=False)

        if return_activations:
            return output, activations
        else:
            return output
