from functools import lru_cache
from typing import Literal

import torch
from torch import nn


class CMIFNormalization:
    def computeEntropies(self, Px, Py, Pxy):
        return (
            -(Px * Px.log()).where(Px > 0, 0).sum(-3),
            -(Py * Py.log()).where(Py > 0, 0).sum(-3),
            -(Pxy * Pxy.log()).where(Pxy > 0, 0).sum((-4, -3)),
        )

    def noneNorm(self, Pxy, Px, Py):
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

    def sumNorm(self, Pxy, Px, Py):
        Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
        return 2 * (Hx + Hy - Hxy) / (Hx + Hy)

    def jointNorm(self, Pxy, Px, Py):
        Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
        return (Hx + Hy - Hxy) / Hxy

    def maxNorm(self, Pxy, Px, Py):
        Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
        return (Hx + Hy - Hxy) / torch.max(Hx, Hy)

    def sqrtNorm(self, Pxy, Px, Py):
        Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
        return (Hx + Hy - Hxy) / torch.sqrt(Hx * Hy)

    def minNorm(self, Pxy, Px, Py):
        Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
        return (Hx + Hy - Hxy) / torch.min(Hx, Hy)


class FastCMIF(nn.Module, CMIFNormalization):
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

    def _normalize(self, im):
        minV, maxV = [
            v.unsqueeze(-1) for v in im.flatten(-2, -1).aminmax(dim=(-1), keepdim=True)
        ]
        return (im - minV) / (maxV - minV)

    def _quantify(self, im):
        return (im * (self.nBins - 1)).floor().long()

    def _one_hot(self, im):
        return torch.nn.functional.one_hot(im, self.nBins).moveaxis(-1, -3)

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

        match norm:
            case "none":
                return self.noneNorm
            case "sum":
                return self.sumNorm
            case "joint":
                return self.jointNorm
            case "max":
                return self.maxNorm
            case "sqrt":
                return self.sqrtNorm
            case "min":
                return self.minNorm
            case _:
                raise ValueError(
                    "Normalization must be one of 'none', 'sum', \
'joint', 'max', 'sqrt', 'min'"
                )

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
                        (im.size(-2) + template.size(-2) - template.size(-2) % 2),
                        (im.size(-1) + template.size(-1) - template.size(-1) % 2),
                    )
                ),
            )
            * torch.fft.rfft2(torch.flip(template, dims=(-2, -1)), padded_shape)
        )[
            ...,
            (hH := template.size(-2) // 2) : -(hH),
            (hW := template.size(-1) // 2) : -(hW),
        ]

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
        Pxy = self._fftconv(
            imOh.unsqueeze(-3), templateOh.unsqueeze(-4), *padding
        ) / N.unsqueeze(-3)
        return self.norm(Pxy, Px / N, Py / N)
