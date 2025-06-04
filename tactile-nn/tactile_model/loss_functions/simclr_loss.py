import torch
import torch.nn as nn

from .necks import NonLinearNeckSimCLR
from .heads import ContrastiveHead
from .loss_function_base import LossFunctionBase

__all__ = ["SimCLRLoss"]


class SimCLRLoss(LossFunctionBase):
    """SimCLR.
    Implementation of "A Simple Framework for Contrastive Learning
    of Visual Representations (https://arxiv.org/abs/2002.05709)".
    Args:
        model_output_dim      : (int) output dimension of model without FC layer.
        hidden_dim.           : (int) dimension of hidden layer.
                                Default: None, uses the model_output_dim.
        embedding_dim         : (int) dimension of embedding for the hidden layer.
                                Default: 128.
        temperature           : (float) Temperature parameter of contrastive loss.
                                Default: 0.1.
    """

    def __init__(
        self,
        model_output_dim,
        hidden_dim=2048,
        embedding_dim=128,
        temperature=0.1,
    ):
        super(SimCLRLoss, self).__init__()

        self._hidden_dim = hidden_dim
        self.neck = NonLinearNeckSimCLR(
            in_channels=model_output_dim,
            hid_channels=self._hidden_dim,
            out_channels=embedding_dim,
        )
        self.neck.init_weights()
        self.head = ContrastiveHead(temperature=temperature)

    # def trainable_parameters(self):
    #     return self.neck.parameters()
    #
    # def named_parameters(self, *args, **kwargs):  # using "*args, **kwargs" in case the signature changes
    #     return self.neck.named_parameters(*args, **kwargs)

    def _create_buffer(self, N, device):
        # ensures that these are on the same device as the input
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).to(device)
        pos_ind = (
            torch.arange(N * 2).to(device),
            2
            * torch.arange(N, dtype=torch.long)
            .unsqueeze(1)
            .repeat(1, 2)
            .view(-1, 1)
            .squeeze()
            .to(device),
        )
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).to(device)
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def forward(self, x, **kwargs):
        """Forward computation during training.
        Args:
            x (Tensor): Input of two concatenated images of shape (2*N, d).
                Typically these should be mean centered and std scaled.
        Returns:
            Loss.

        Adapted from: https://github.com/open-mmlab/OpenSelfSup/blob/ed5000482b0d8b816cd8a6fbbb1f97da44916fed/openselfsup/models/simclr.py#L68-L96
        """
        if not isinstance(x, list):
            x = [x]
        z = self.neck(x)[0]  # (2n)xd
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)

        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N, device=z.device)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        return losses


if __name__ == "__main__":
    loss_func = SimCLRLoss(model_output_dim=512)

    inputs = torch.rand(40, 512)
    loss = loss_func(x=inputs)
