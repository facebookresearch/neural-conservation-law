"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn, Tensor
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x


def softplus(x):
    return nn.functional.softplus(x)


def gaussian_softplus(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-(x ** 2) / 2) + z * x) / (
        2 * z
    )


def gaussian_softplus2(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-(x ** 2) / 2) + z * x) / z


def laplace_softplus(x):
    return torch.relu(x) + torch.exp(-torch.abs(x)) / 2


def cauchy_softplus(x):
    # (Pi y + 2 y ArcTan[y] - Log[1 + y ^ 2]) / (2 Pi)
    pi = np.pi
    return (x * pi - torch.log(x ** 2 + 1) + 2 * x * torch.atan(x)) / (2 * pi)


def activation_shifting(activation):
    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))

    return shifted_activation


def get_softplus(softplus_type="softplus", zero_softplus=False):
    if softplus_type == "softplus":
        act = nn.functional.softplus
    elif softplus_type == "gaussian_softplus":
        act = gaussian_softplus
    elif softplus_type == "gaussian_softplus2":
        act = gaussian_softplus2
    elif softplus_type == "laplace_softplus":
        act = gaussian_softplus
    elif softplus_type == "cauchy_softplus":
        act = cauchy_softplus
    else:
        raise NotImplementedError(f"softplus type {softplus_type} not supported.")
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):
    def __init__(self, softplus_type="softplus", zero_softplus=False):
        super(Softplus, self).__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return (
            nn.functional.linear(
                x, torch.nn.functional.softplus(self.weight), self.bias
            )
            * gain
        )


class ICNN3(torch.nn.Module):
    def __init__(
        self,
        dim=2,
        dimh=16,
        num_hidden_layers=2,
        symm_act_first=False,
        softplus_type="softplus",
        zero_softplus=False,
    ):
        super(ICNN3, self).__init__()

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh // 2, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh // 2))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(nn.Linear(dim, dimh // 2))
        self.Wx2s = torch.nn.ModuleList(Wx2s)

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.Wzs[0](x), self.act)
        else:
            z = self.act(self.Wzs[0](x))
        for Wz, Wx, Wx2 in zip(self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s[:],):
            z = self.act(Wz(z) + Wx(x))
            aug = Wx2(x)
            aug = symm_softplus(aug, self.act) if self.symm_act_first else self.act(aug)
            z = torch.cat([z, aug], 1)
        return self.Wzs[-1](z) + self.Wxs[-1](x)


class PICNN(nn.Module):
    def __init__(
        self,
        dim=2,
        dimh=16,
        dimc=2,
        num_hidden_layers=2,
        PosLin=PosLinear,
        symm_act_first=False,
        softplus_type="gaussian_softplus",
        zero_softplus=False,
    ):
        super(PICNN, self).__init__()

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.act_c = nn.ELU()
        self.symm_act_first = symm_act_first

        # data path
        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLin(dimh, dimh, bias=True))
        Wzs.append(PosLin(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        # skip data
        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        # context path
        Wcs = list()
        Wcs.append(nn.Linear(dimc, dimh))
        self.Wcs = torch.nn.ModuleList(Wcs)

        Wczs = list()
        for _ in range(num_hidden_layers - 1):
            Wczs.append(nn.Linear(dimh, dimh))
        Wczs.append(nn.Linear(dimh, dimh, bias=True))
        self.Wczs = torch.nn.ModuleList(Wczs)
        for Wcz in self.Wczs:
            Wcz.weight.data.zero_()
            Wcz.bias.data.zero_()

        Wcxs = list()
        for _ in range(num_hidden_layers - 1):
            Wcxs.append(nn.Linear(dimh, dim))
        Wcxs.append(nn.Linear(dimh, dim, bias=True))
        self.Wcxs = torch.nn.ModuleList(Wcxs)
        for Wcx in self.Wcxs:
            Wcx.weight.data.zero_()
            Wcx.bias.data.zero_()

        Wccs = list()
        for _ in range(num_hidden_layers - 1):
            Wccs.append(nn.Linear(dimh, dimh))
        self.Wccs = torch.nn.ModuleList(Wccs)

    def forward(self, x, c):
        if self.symm_act_first:
            z = symm_softplus(self.Wzs[0](x), self.act)
        else:
            z = self.act(self.Wzs[0](x))
        c = self.act_c(self.Wcs[0](c))
        for Wz, Wx, Wcz, Wcx, Wcc in zip(
            self.Wzs[1:-1], self.Wxs[:-1], self.Wczs[:-1], self.Wcxs[:-1], self.Wccs
        ):
            cz = softplus(Wcz(c) + np.exp(np.log(1.0) - 1))
            cx = Wcx(c) + 1.0
            z = self.act(Wz(z * cz) + Wx(x * cx) + Wcc(c))

        cz = softplus(self.Wczs[-1](c) + np.log(np.exp(1.0) - 1))
        cx = self.Wcxs[-1](c) + 1.0
        return self.Wzs[-1](z * cz) + self.Wxs[-1](x * cx)


def test_picnn():
    import matplotlib.pyplot as plt

    print("Testing convexity")
    n = 64
    dim = 123
    dimh = 16
    dimc = 11
    num_hidden_layers = 2
    picnn = PICNN(dim=dim, dimh=dimh, dimc=dimc, num_hidden_layers=num_hidden_layers)
    x1 = torch.randn(n, dim)
    x2 = torch.randn(n, dim)
    c = torch.randn(n, dimc)
    print(
        np.all(
            (((picnn(x1, c) + picnn(x2, c)) / 2 - picnn((x1 + x2) / 2, c)) > 0)
            .cpu()
            .data.numpy()
        )
    )

    print("Visualizing convexity")
    dim = 1
    dimh = 16
    dimc = 1
    num_hidden_layers = 2
    picnn = PICNN(dim=dim, dimh=dimh, dimc=dimc, num_hidden_layers=num_hidden_layers)

    c = torch.zeros(1, dimc)
    x = torch.linspace(-10, 10, 100).view(100, 1)
    for c_ in np.linspace(-5, 5, 10):
        plt.plot(x.squeeze().numpy(), picnn(x, c + c_).squeeze().data.numpy())
    plt.savefig("picnn.png")


def test_icnn():
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    icnn = ICNN3(dim=1, dimh=128, num_hidden_layers=3).to(device)
    x = torch.linspace(-10, 10, 1000).reshape(-1, 1).to(device).requires_grad_(True)
    y = icnn(x)

    dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    ddx = torch.autograd.grad(dx.sum(), x)[0]

    plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    plt.plot(x.detach().cpu().numpy(), ddx.detach().cpu().numpy(), "--")
    plt.savefig("icnn.png")


if __name__ == "__main__":
    test_icnn()
    test_picnn()
