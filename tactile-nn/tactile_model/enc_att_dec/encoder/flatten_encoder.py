import torch.nn as nn
from pt_tnn.pre_post_memory import FullyConnected
import einops

__all__ = ["FlattenEncoder"]


class FlattenEncoder(nn.Module):
    def __init__(self, cfg):
        super(FlattenEncoder, self).__init__()
        self.cfg = cfg

        repr_dim = cfg.model.repr_dim  # output dim of the encoder

        self.lin = FullyConnected(in_channels=0,  # serve as a placeholder
                                  out_channels=repr_dim,
                                  init_dict=None,
                                  dropout=0.0,
                                  use_bias=True,
                                  bias=None,  # 0.1
                                  batchnorm=None,
                                  activation=None
                                  )  # a lazy linear layer

    def forward(self, x, return_activations=False):
        activations = dict()
        shape = x.shape  # (b, t, C, H, W) or (b, C, H, W)
        # rearrange
        if len(shape) == 5:  # has time dimension, later a non-identity attender will be used
            x = einops.rearrange(x, "b t c h w -> b t (c h w)")
            x = self.lin(x)
            activations['enc_flat_enc'] = x
        else:
            print(f'the input shape={shape} is not supported')
            raise Exception
        if return_activations:
            return x, activations
        else:
            return x
