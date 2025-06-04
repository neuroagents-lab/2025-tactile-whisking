import torch.nn as nn
from pt_tnn.pre_post_memory import FullyConnected
import einops
import omegaconf

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.cfg = cfg

        self.decoder_cfg = cfg.model.decoder

        self.n_layer = self.decoder_cfg.n_layer
        self.out_channels = self.decoder_cfg.out_channels
        self.init_dict = self.decoder_cfg.init_dict
        self.dropout = self.decoder_cfg.dropout
        self.use_bias = self.decoder_cfg.use_bias
        self.bias = self.decoder_cfg.bias
        self.batchnorm = self.decoder_cfg.batchnorm
        self.activation = self.decoder_cfg.activation

        self.temporal_decoder = self.decoder_cfg.temporal_decoder
        self.temporal_steps = self.decoder_cfg.temporal_steps

        if self.temporal_steps == 'all':
            print('using all timesteps of (bs, t, d) in the MLP decoder')
        elif isinstance(self.temporal_steps, omegaconf.listconfig.ListConfig):
            self.temporal_steps = list(self.temporal_steps)
            print(f'using timesteps {self.temporal_steps} of (bs, t, d) in the MLP decoder')
        else:
            print('the temporal_steps is neither "all" nor a list from the decoder cfg')
            raise Exception

        assert len(self.out_channels) == self.n_layer

        if self.n_layer > 0:  # when it's 0, it will be the dummy decoder for SSL models
            self.out_channels[-1] = self.cfg.model.out_dim  # override the output dim of the last linear layer

        fc_layers = []
        for idx, out_channels in enumerate(self.out_channels):
            layer = FullyConnected(in_channels=0,  # serve as a placeholder
                                   out_channels=out_channels,
                                   init_dict=self.init_dict,  # e.g., {"method": "trunc_normal_"}
                                   dropout=self.dropout,
                                   use_bias=self.use_bias,
                                   bias=self.bias,  # 0.1
                                   batchnorm=self.batchnorm,
                                   activation=None if idx == self.n_layer - 1 else self.activation
                                   # mute the activation in the last layer
                                   )

            fc_layers.append(layer)

        self.mlp = nn.Sequential(*fc_layers)
        print(self.mlp)

    def forward(self, x, return_activations=False):
        # x: (bs, T, d) or (bs, d)
        if self.temporal_decoder:
            if self.temporal_steps != 'all':
                x = x[:, self.temporal_steps, :]
            x = einops.rearrange(x, 'b t d -> b (t d)')

        # mlp will sort anything (bs, d1, d2, ...) into (bs, d1*d2*...)
        if return_activations:
            activations = dict()
            for idx, layer in enumerate(self.mlp):
                x = layer(x)
                activations[f'dec_mlp_l{idx}'] = x
            return x, activations
        else:
            return self.mlp(x)
