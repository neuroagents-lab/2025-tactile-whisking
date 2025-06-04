import torch
import torch.nn as nn
from pt_tnn.temporal_graph import TemporalGraph
from pt_tnn.pre_post_memory import FullyConnected
import json
from utils import input_transform
from omegaconf import OmegaConf


class TactileTNN(nn.Module):
    """
    Definition of the computation graph, including reading the graph from the
    configuration file and unrolling the graph in time.
    """

    def __init__(self, cfg):
        super(TactileTNN, self).__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train
        self.test_cfg = cfg.test

        self.n_times = self.train_cfg.n_times
        self.full_unroll = self.train_cfg.full_unroll

        config = self.override_model_config()

        transform = self.train_cfg.get("input_transform", None)
        if transform is None or transform.name is None:
            self.transform = None
        else:
            self.transform = getattr(input_transform, transform.name)(**transform.kwargs)

        self.TG = TemporalGraph(model_config_file=config,  # [path] self.model_cfg.model_config_file,
                                # recurrent_module=TactileRecurrentModule,
                                input_shape=self.data_cfg.input_shape, num_timesteps=self.n_times,
                                transform=self.transform)
        # override the `input_shape` and `num_timesteps` from the model_config_file

        if 'addnet' in config.keys():
            fc_layers = []
            for i, layer_id in enumerate(config['addnet'].keys()):
                out_channels = config['addnet'][layer_id]['fc']['num_features']
                batchnorm = config['addnet'][layer_id]['batchnorm'] \
                    if 'batchnorm' in config['addnet'][layer_id].keys() else None
                activation = None if i == len(config['addnet'].keys()) - 1 else 'ReLU'

                layer = FullyConnected(in_channels=0,  # serve as a placeholder
                                       out_channels=out_channels,
                                       init_dict=None,  # {"method": "trunc_normal_"}
                                       dropout=self.train_cfg.dropout,
                                       use_bias=True,
                                       bias=None,  # 0.1
                                       batchnorm=batchnorm,
                                       activation=activation)

                fc_layers.append(layer)

            self.add_net = nn.Sequential(*fc_layers)
        else:
            self.add_net = nn.Identity()

    def override_model_config(self):
        with open(self.model_cfg.model_config_file, "r") as f:
            config = json.load(f)

        # override the output channel in the last layer
        if 'addnet' in config.keys():
            # has addnet (e.g., Zhuang's model)
            last_layer_id = list(config['addnet'].keys())[-1]
            config['addnet'][last_layer_id]['fc']['num_features'] = self.data_cfg.num_classes
        else:
            # no addnet in the config (e.g., ResNet18, AlexNet)
            output_node = config["output_node"]
            output_node_config = None
            for node_config in config["nodes"]:
                if node_config["name"] == output_node:
                    output_node_config = node_config

            output_node_config["out_channels"] = self.data_cfg.num_classes

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
            modified = modify_out_channels(mem=output_post_mem, num_classes=self.data_cfg.num_classes)
            if not modified:  # then try to find prediction head in pre_mem
                modified = modify_out_channels(mem=output_pre_mem, num_classes=self.data_cfg.num_classes)

            if not modified:
                print('no prediction layers are found in the output node config, please make sure the model is correct')
                print('below is the modified version of the output node:')
                print(output_node_config)
                raise Exception
        # print(config)
        return config

    def forward(self, x, cuda=False):

        if self.full_unroll:
            output = self.TG.full_unroll(x, n_times=self.n_times, cuda=cuda)
        else:
            output = self.TG(x, n_times=self.n_times, cuda=cuda)

        return self.add_net(output)
