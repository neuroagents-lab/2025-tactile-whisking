import torch.nn as nn
from torchvision.models import resnet18
import einops

__all__ = ["PTResNet18"]


class PTResNet18(nn.Module):
    def __init__(self, cfg):
        super(PTResNet18, self).__init__()

        self.cfg = cfg
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data

        self.repr_dim = self.model_cfg.repr_dim

        C, H, W = self.data_cfg.input_shape

        self.encoder = resnet18()
        self.encoder.conv1 = nn.Conv2d(C, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.fc = nn.Linear(in_features=self.encoder.fc.in_features, out_features=self.repr_dim)

        self.activations = {}
        self._register_activation_hooks()

    def _register_activation_hooks(self):
        """
        Registers a forward hook on each submodule of self.encoder,
        storing outputs in self.activations.
        """

        def hook_fn(name):
            def forward_hook(module, input, output):
                self.activations[name] = output

            return forward_hook

        # Register on all submodules (or selectively filter if you want)
        for name, module in self.encoder.named_modules():
            if name in ["relu", "maxpool", "layer1.0", "layer1.1", "layer2.0", "layer2.1",
                        "layer3.0", "layer3.1", "layer4.0", "layer4.1", "avgpool", "fc"]:
                module.register_forward_hook(hook_fn(name))

    def forward(self, x, return_activations=False):
        shape = x.shape  # (b, t, C, H, W) or (b, C, H, W)

        # Clear out any old activations before this forward
        self.activations.clear()

        # rearrange
        if len(shape) == 5:  # has time dimension, later a non-identity attender will be used
            x = einops.rearrange(x, "b t c h w -> (b t) c h w")
            x = self.encoder(x)  # (b*t, [C, H, W] ---> d)
            x = einops.rearrange(x, "(b t) d -> b t d", t=shape[1])
        elif len(shape) == 4:  # (b, C, H, W), later an identity attender will be used
            x = self.encoder(x)  # (b, d)
        else:
            print(f'the input shape={shape} is not supported')
            raise Exception

        if return_activations:
            return x, self.activations
        else:
            return x
