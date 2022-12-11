"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from icnn import PICNN
from subharmonic import SubharmonicMixture


class NeuralConservationLaw(nn.Module):
    def __init__(self, dim, d_model, num_hidden_layers=4, n_mixtures=128, actfn="swish"):
        super().__init__()

        if dim - 1 == 1:
            self.F_zero = PICNN(
                dim=dim - 1, dimh=d_model, dimc=1, num_hidden_layers=num_hidden_layers
            )
        else:
            self.F_zero = SubharmonicMixture(
                dim=dim - 1,
                d_model=d_model,
                num_hidden_layers=num_hidden_layers,
                n_mixtures=n_mixtures,
            )

        layers = [
            nn.Linear(dim - 1, d_model),
            nn.Softplus(beta=20),
        ]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.Softplus(beta=20))
        layers.append(nn.Linear(d_model, dim - 1))

        self.F_other = nn.Sequential(*layers)

    def forward(self, state):
        if state.ndim == 1:
            F0 = self.F_zero(state[None, 1:], state[None, 0:1])[0]
        else:
            F0 = self.F_zero(state[:, 1:], state[:, 0:1])

        x = state[..., 1:]
        F1D = self.F_other(x)
        F = torch.cat([F0, F1D], dim=-1)
        return F


class Swish(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5] * dim))

    def forward(self, x):
        return x * torch.sigmoid_(x * F.softplus(self.beta))

    def extra_repr(self):
        return f"{self.beta.nelement()}"
