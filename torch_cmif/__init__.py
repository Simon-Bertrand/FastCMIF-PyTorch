from functools import lru_cache
from typing import Literal
import torch

from torch import nn


class FastCMIF(nn.Module):
    def __init__(
        self,
        nBins: int,
        norm: Literal["none", "sum", "joint", "max", "sqrt", "min"] = "none",
    ) -> None:
        """
        Initialize the TorchCMIF module.

        Args:
            nBins (int): The number of bins for the histogram.
            norm (str, optional): The normalization method for the histogram.
            Defaults to "none".

        Returns:
            None
        """
        super().__init__()
        self.nBins = nBins
        self.norm = self._chooseNorm(norm)

    @staticmethod
    def findArgmax(x):
        """
        Finds the indices of the maximum values along the last dimension of
        the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The indices of the maximum values.
        """
        aMax = x.flatten(-2, -1).argmax(dim=-1)
        return torch.stack([aMax // x.size(-1), aMax % x.size(-1)])

    def _normalize(self, im):
        minV, maxV = [v.unsqueeze(-1) for v in im.flatten(-2, -1).aminmax(dim=(-1), keepdim=True)]
        return (im - minV) / (maxV - minV)

    def _quantify(self, im):
        return (im * (self.nBins - 1)).floor().long()

    def _one_hot(self, im):
        return torch.nn.functional.one_hot(im, self.nBins).moveaxis(-1, -3)

    @torch.compile
    def _chooseNorm(self, norm: Literal["none", "sum", "joint", "max", "sqrt", "min"]):
        """
        Chooses the normalization function based on the given normalization
        type.

        Args:
            norm (Literal["none", "sum", "joint", "max", "sqrt", "min"]):
            The normalization type.

        Returns:
            function: The normalization function.

        Raises:
            ValueError: If the normalization type is not valid.

        """

        def computeEntropies(Px, Py, Pxy):
            return (
                -(Px * Px.log()).where(Px > 0, 0).sum(-3),
                -(Py * Py.log()).where(Py > 0, 0).sum(-3),
                -(Pxy * Pxy.log()).where(Pxy > 0, 0).sum((-4, -3)),
            )

        def noneNorm(Pxy, Px, Py):
            PxPy = Px.unsqueeze(-3) * Py.unsqueeze(-4)
            return (
                torch.nn.functional.kl_div(
                    PxPy.log(),
                    Pxy,
                    reduction="none",
                )
                .where((Pxy > 0) & (PxPy > 0), 0)
                .sum(dim=(-4, -3))
            )

        def sumNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = computeEntropies(Px, Py, Pxy)
            return 2 * (Hx + Hy - Hxy) / (Hx + Hy)

        def jointNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = computeEntropies(Px, Py, Pxy)
            return (Hx + Hy - Hxy) / Hxy

        def maxNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = computeEntropies(Px, Py, Pxy)
            return (Hx + Hy - Hxy) / torch.max(Hx, Hy)

        def sqrtNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = computeEntropies(Px, Py, Pxy)
            return (Hx + Hy - Hxy) / torch.sqrt(Hx * Hy)

        def minNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = computeEntropies(Px, Py, Pxy)
            return (Hx + Hy - Hxy) / torch.min(Hx, Hy)

        match norm:
            case "none":
                return noneNorm
            case "sum":
                return sumNorm
            case "joint":
                return jointNorm
            case "max":
                return maxNorm
            case "sqrt":
                return sqrtNorm
            case "min":
                return minNorm
            case _:
                raise ValueError(
                    "Normalization must be one of 'none', 'sum', \
'joint', 'max', 'sqrt', 'min'"
                )

    @lru_cache(maxsize=2)
    def _nextFastLen(self, size):
        """
        Computes the next fast length for FFT-based method.

        Args:
            size (int): The current size.

        Returns:
            int: The next fast length.
        """
        next_size = size
        while True:
            remaining = next_size
            for n in (2, 3, 5):
                while (euclDiv := divmod(remaining, n))[1] == 0:
                    remaining = euclDiv[0]
            if remaining == 1:
                return next_size
            next_size += 1

    def _fftconv(self, im, template, padWl, padHt):
        """
        Performs convolution using Fast Fourier Transform (FFT).

        Args:
            im (torch.Tensor): Input image tensor (B,C,H,W).
            template (torch.Tensor): Template tensor (B,C,h,w).
            padWl (int): Width left padding.
            padHt (int): Height top padding.

        Returns:
            torch.Tensor: Convolved output tensor.
        """
        return torch.fft.irfft2(
            torch.fft.rfft2(
                im,
                s=(
                    padded_shape := (
                        self._nextFastLen(im.size(-2) + template.size(-2) - 1),
                        self._nextFastLen(im.size(-1) + template.size(-1) - 1),
                    )
                ),
            )
            * torch.fft.rfft2(torch.flip(template, dims=(-1, -2)), padded_shape)
        )[
            ...,
            padHt : padHt + im.size(-2),
            padWl : padWl + im.size(-1),
        ]

    def forward(self, im, template):
        """
        Performs forward pass of the model.

        Args:
            im (torch.Tensor): Input image tensor.
            template (torch.Tensor): Template tensor.

        Returns:
            torch.Tensor: Result of the forward pass.
        """
        padding = [
            (template.size(-1) - 1) // 2,
            ((template.size(-2) - 1) // 2),
        ]
        imOh = self._one_hot(self._quantify(self._normalize(im)))
        templateOh = self._one_hot(self._quantify(self._normalize(template)))
        Px, Py = self._fftconv(
            torch.stack([imOh, torch.ones_like(imOh)]),
            torch.stack([torch.ones_like(templateOh), templateOh]),
            *padding,
        )
        N = Px.sum(-3, keepdim=True)
        Pxy = self._fftconv(imOh.unsqueeze(-3), templateOh.unsqueeze(-4), *padding) / N.unsqueeze(
            -3
        )
        return self.norm(Pxy, Px / N, Py / N)
