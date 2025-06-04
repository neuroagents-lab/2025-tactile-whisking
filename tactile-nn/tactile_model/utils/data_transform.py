import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["ShapeNetReshape"]


class ShapeNetReshape(nn.Module):
    """
    A custom transform that reshapes the tensor from ShapeNet as (110, 35, 6) ---> (22, 30, 5, 7)
    """

    def __init__(self, n_times: int):
        super(ShapeNetReshape, self).__init__()
        self.n_times = n_times

    def forward(self, data: torch.Tensor):
        data = rearrange(data, "(ntime step) (H W) f -> ntime (step f) H W", ntime=self.n_times, H=5, W=7)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(n_times={self.n_times})'
