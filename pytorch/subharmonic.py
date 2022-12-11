"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubharmonicMixture(nn.Module):
    """Constructs a function whose Laplacian is non-negative
    by adding up a bunch of functions whose Laplacian is
    the unnormalized density of an isometric Gaussian.
    """

    def __init__(self, dim, d_model, num_hidden_layers, n_mixtures):
        super().__init__()
        self.dim = dim
        self.n_mixtures = n_mixtures

        layers = [nn.Linear(1, d_model), Swish(d_model)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(Swish(d_model))
        layers.append(nn.Linear(d_model, n_mixtures * (3 + dim)))
        self.params = nn.Sequential(*layers)

        self.out_scale = nn.Parameter(torch.randn(1) * 0.2)
        self.exp1 = Exp1()

    def forward(self, x, t):
        shape = x.shape[:-1]
        D = x.shape[-1]

        dtype = x.dtype

        if D < 2:
            raise RuntimeError("Dimensions less than 2 are not supported.")

        params = self.params(t)
        params = params.reshape(*shape, self.n_mixtures, 3 + self.dim).double()
        logits, scale_in, scale_out, means = torch.split(
            params, [1, 1, 1, self.dim], dim=-1
        )
        weights = torch.softmax(logits.squeeze(-1), dim=-1)

        x = x.reshape(*shape, 1, self.dim)
        x = x.double()

        # scatter the means at initialization
        x = x * F.softplus(scale_in + 1.0) + means * 10

        r_sq = (x * x).sum(dim=-1)  # (..., n_mixtures)

        # if D == 2:
        exp1 = self.exp1(r_sq / 2)
        out = 1 / (4 * np.pi) * (torch.log(r_sq) + exp1)
        out = out * F.softplus(scale_out.squeeze(-1))
        # else:
        #     r = (r_sq + 1e-5).pow(0.5)
        #     out = torch.erf(r / np.sqrt(2))
        #     out = out.mul(-1 / (4 * np.pi) / r)
        #     out = out * F.softplus(scales).squeeze(-1)

        return (out * weights).sum(dim=-1, keepdim=True).to(dtype)


class Swish(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5] * dim))

    def forward(self, x):
        return x * torch.sigmoid_(x * F.softplus(self.beta))

    def extra_repr(self):
        return f"{self.beta.nelement()}"


class Exp1(nn.Module):
    """Uses the Alan and Hasting approximation to compute exp1."""

    def __init__(self):
        super().__init__()
        self.a = torch.tensor([-0.57722, 0.99999, -0.24991, 0.05519, -0.00976, 0.00108])
        self.b = torch.tensor([0.26777, 8.63476, 18.05902, 8.57333])
        self.c = torch.tensor([3.95850, 21.09965, 25.63296, 9.57332])

    def _xk(self, x, k):
        return x.unsqueeze(-1).pow(torch.arange(k + 1).to(x))

    def forward(self, x):
        a = self.a.to(x)
        b = self.b.to(x)
        c = self.c.to(x)

        x5 = self._xk(x, 5)
        x3 = self._xk(x, 3)
        below_one = -torch.log(x) + torch.sum(a * x5, dim=-1)
        above_one = (
            torch.exp(-x) / x * torch.sum(b * x3, dim=-1) / torch.sum(c * x3, dim=-1)
        )
        return torch.where(x <= 1, below_one, above_one)
