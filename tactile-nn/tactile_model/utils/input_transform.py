import torch
import torch.nn as nn
from torchvision.transforms import *
from einops import rearrange

__all__ = ["MakeMovie", "SSLTransform"]


class MakeMovie(nn.Module):
    """
    A custom transform that repeats the given tensor for `image_off` frames,
    then inserts blank frames (filled with 0.5) for the remaining `times - image_off` frames.

    Args:
        times (int): Total number of frames.
        image_off (int): Number of frames to show the original image/tensor before switching to blank frames.
    """

    def __init__(self, times: int, image_off: int):
        super(MakeMovie, self).__init__()
        self.times = times
        self.image_off = image_off

    def forward(self, ims: torch.Tensor):
        """
        Args:
            ims (torch.Tensor): The input tensor (e.g., a single image or batch of images).

        Returns:
            torch.Tensor: bs, t=self.image_off+(self.times - self.image_off), ...
        """
        shape = ims.shape
        if len(shape) == 4:
            blank = torch.full_like(ims, 0.5)
            pres = [ims] * self.image_off + [blank] * (self.times - self.image_off)
            return torch.stack(pres, dim=1)
        elif len(shape) == 5:
            assert shape[1] == self.times, f'T={self.times} but got input time dimension={shape[1]}'
            ims[:, self.image_off:, ...] = 0.5
            return ims
        else:
            print(f'currently only (bs, C, H, W) or (bs, T, C, H, W) is supported, but the input shape={shape}')
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + f'(times={self.times}-image-off={self.image_off})'


class RandomTemporalFlip(nn.Module):
    """Temporal flip the given image randomly with a given probability.

    Args:
        dims (list[ints]): the dimensions to be flipped.
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, dims: list[int], p=0.5):
        super().__init__()
        self.dims = dims
        self.p = p

    def forward(self, inp):
        """
        Args:
            inp (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return -torch.flip(inp, dims=self.dims)
        return inp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims}-p={self.p})"


class Random90Rotation:
    """Rotate a 4D or 3D tensor by 90, 180 or 270 degrees on its last two dims."""

    def __init__(self, multipliers=(1, 2, 3)):
        """angles: tuple of allowed rotation angles."""
        self.transform = RandomChoice(
            [RandomRotation(degrees=(90 * k, 90 * k)) for k in multipliers])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is (bs, T, C, H, W)
        We'll rotate the last two dimensions.
        """
        ntime = x.shape[1]
        x = rearrange(x, "bs ntime C H W -> bs (ntime C) H W", ntime=ntime, H=5, W=7)
        x = self.transform(x)
        x = rearrange(x, "bs (ntime C) H W -> bs ntime C H W", ntime=ntime, H=5, W=7)
        return x


class SSLTransform(nn.Module):
    def __init__(self, use_temporal):
        super(SSLTransform, self).__init__()
        self.use_temporal = use_temporal

        self.hflip = RandomHorizontalFlip(p=0.5)
        self.vflip = RandomVerticalFlip(p=0.5)

        self.rot = Random90Rotation(multipliers=(1, 2, 3, 4))

        if self.use_temporal:
            self.tflip = RandomTemporalFlip(dims=[1], p=0.5)

    def forward(self, inp):
        """
        Args:
            inp (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < 0.5:
            inp = self.hflip(inp)
        else:
            inp = self.vflip(inp)

        inp = self.rot(inp)

        if self.use_temporal:
            # inp should be of shape (bs, 22, 30, 5, 7), as suggested by the following data transform:
            ntime = inp.shape[1]
            inp = rearrange(inp, "bs ntime (step f) H W -> bs (ntime step) f H W", f=6, H=5, W=7)
            inp = self.tflip(inp)
            inp = rearrange(inp, "bs (ntime step) f H W -> bs ntime (step f) H W", ntime=ntime, H=5, W=7)

        return inp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(use_temporal={self.use_temporal})"


# https://github.com/neuroailab/mouse-vision/blob/5c1e85dfd2cc805873f0e6c71d11d070f3fadf6f/mouse_vision/model_training/trainer_transforms.py#L413

class SimCLRTransform(nn.Module):
    def __init__(self, **kwargs):
        super(SimCLRTransform, self).__init__()

        self.img_transform = Compose([
            # 2. Random horizontal flip with p=0.5 (default)
            RandomHorizontalFlip(),

            # 3. Color jitter applied with prob. 0.8
            RandomApply([ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                        p=0.8
                        ),

            # 4. Convert to gray‑scale with prob. 0.2
            RandomGrayscale(p=0.2),

            # 5. Gaussian blur with prob. 0.5
            RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),

        ])

    def forward(self, inp):
        """
        Args:
            inp (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        bs, ntime = inp.shape[:2]
        inp = rearrange(inp, "bs ntime (res C) H W -> (bs ntime res) C H W", C=3, H=5, W=7)
        inp = self.img_transform(inp)
        inp = rearrange(inp, "(bs ntime res) C H W -> bs ntime (res C) H W", bs=bs, ntime=ntime, C=3)

        return inp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class SimSiamTransform(nn.Module):
    def __init__(self, **kwargs):
        super(SimSiamTransform, self).__init__()

        self.img_transform = Compose([
            # 2. Random horizontal flip with p=0.5 (default)
            RandomHorizontalFlip(),

            # 3. Color jitter applied with prob. 0.8
            RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                        p=0.8
                        ),

            # 4. Convert to gray‑scale with prob. 0.2
            RandomGrayscale(p=0.2),

            # 5. Gaussian blur with prob. 0.5
            RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),

        ])

    def forward(self, inp):
        """
        Args:
            inp (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        bs, ntime = inp.shape[:2]
        inp = rearrange(inp, "bs ntime (res C) H W -> (bs ntime res) C H W", C=3, H=5, W=7)
        inp = self.img_transform(inp)
        inp = rearrange(inp, "(bs ntime res) C H W -> bs ntime (res C) H W", bs=bs, ntime=ntime, C=3)

        return inp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
