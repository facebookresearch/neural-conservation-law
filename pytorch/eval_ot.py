"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import click
import pickle as pkl
import numpy as np
import torch
from functools import partial
from functorch import vmap

import matplotlib.pyplot as plt

from dist2d import load_2dtarget
from main_ot import Workspace as W


@click.group()
def cli():
    pass


def plt_vecfield(vecfield_fn, ax, npts=20, device="cpu"):
    side = np.linspace(-2, 2, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    v = vecfield_fn(x)
    v = v.cpu().numpy().reshape(npts, npts, 2)

    ax.quiver(xx, yy, v[:, :, 0], v[:, :, 1])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


@cli.command()
@click.argument("chkpt")
@torch.no_grad()
def plot_vec(chkpt):
    with open(chkpt, "rb") as f:
        workspace = pkl.load(f)

    u_fn = workspace.initialize()
    u_fn = vmap(u_fn, in_dims=(None, 0))

    def vecfield_fn(x, t):
        y = torch.cat([torch.ones(x.shape[0], 1).to(x) * t, x], dim=1)
        u = u_fn(workspace.params, y)
        v = u[..., 1:] / u[..., 0:1].clamp(min=1e-8)

        print(u[..., 0].min(), u[..., 0].max())
        return v

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
    plt_vecfield(partial(vecfield_fn, t=0.00), axs[0], device=workspace.device)
    plt_vecfield(partial(vecfield_fn, t=0.25), axs[1], device=workspace.device)
    plt_vecfield(partial(vecfield_fn, t=0.50), axs[2], device=workspace.device)
    plt_vecfield(partial(vecfield_fn, t=0.75), axs[3], device=workspace.device)
    plt_vecfield(partial(vecfield_fn, t=0.99), axs[4], device=workspace.device)
    fig.tight_layout()
    plt.savefig(f"vecfield.png")
    plt.close()


if __name__ == "__main__":
    cli()

