import torch
import torch.nn as nn
import numpy as np

from .loss_function_base import LossFunctionBase

__all__ = ["AutoEncoderLoss"]


class AutoEncoderLoss(LossFunctionBase):
    def __init__(self, model_output_dim, C, H, W, num_deconv=3, reduction="mean", l1_weighting=1e-4):
        super(AutoEncoderLoss, self).__init__()
        self.output_loss = nn.MSELoss(reduction=reduction)
        self.l1_weighting = l1_weighting
        print(f"Using L1 weighting of {self.l1_weighting}")

        self.decoder = ReverseConvDecoder(d=model_output_dim, C=C, H=H, W=W, num_deconv=num_deconv)

    def forward(self, hidden_vec, inp, **kwargs):
        output = self.decoder(hidden_vec)

        l2_loss = self.output_loss(output, inp)
        l1_reg = torch.sum(torch.abs(hidden_vec))
        loss = 0.5 * l2_loss + (self.l1_weighting / np.prod(hidden_vec.shape)) * l1_reg

        return loss


class ReverseConvDecoder(nn.Module):
    def __init__(self, d, C, H, W, num_deconv=3):
        """
        d: size of the feature dimension that comes in (e.g. 128, 256, etc.)
        C, H, W: target shape per time step; final output should be (bs, T, C, H, W).
        """
        super().__init__()
        self.C = C
        self.H = H
        self.W = W

        channels = [self.C for _ in range(num_deconv + 1)]
        channels[0] = min(self.C, int(d/(self.H*self.W))+1)

        self.fc = nn.Linear(d, channels[0] * self.H * self.W)

        deconv_ops = []
        for idx in range(num_deconv):
            conv_t = nn.ConvTranspose2d(
                in_channels=channels[idx],  # old channels
                out_channels=channels[idx + 1],  # new channels
                kernel_size=3,
                stride=1,
                padding=1
            )
            deconv_ops.append(conv_t)

            if idx != num_deconv - 1:
                deconv_ops.append(nn.ReLU())

        self.deconv = nn.Sequential(*deconv_ops)

    def forward(self, x):
        """
        x: (bs, T*d)
        """
        bs, T, d = x.shape
        # 1) Flatten batch & time into a single dimension
        x = x.view(bs * T, d)  # (bs*T, d)

        # 2) Map from (bs*T, d) -> (bs*T, C-init*H*W), then reshape
        x = self.fc(x).view(bs * T, -1, self.H, self.W)  # (bs*T, C-init*H*W) ---> (bs*T, C-init, H, W)

        # 3) Apply transposed conv(s)
        x = self.deconv(x)

        # 4) Un-flatten time dimension
        x = x.view(bs, T, self.C, x.shape[-2], x.shape[-1])

        return x
