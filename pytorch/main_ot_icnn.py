"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from omegaconf import OmegaConf
import hydra

from functools import partial
import git
import time
import logging
import os
import random
import numpy as np
import pickle as pkl
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from functorch import grad
from torchdiffeq import odeint

from dist2d import load_2dtarget
from dist2d import GaussianMM
from icnn2 import ICNN
import utils


log = logging.getLogger(__name__)


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def plt_density(density_fn, ax, npts=100, device="cpu"):
    side = np.linspace(-2, 2, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    px = density_fn(x)
    px = px.cpu().numpy().reshape(npts, npts)

    ax.imshow(px, extent=[-2, 2, -2, 2], origin="lower")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


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


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)
        log.info(f"workspace: {self.work_dir}")

        self.iter = 0

    def run(self):

        repo = git.Repo(self.file_dir, search_parent_directories=True)
        sha = repo.head.object.hexsha
        log.info(repo)
        log.info(f"Latest commit is {sha}")
        log.info(f"Files modified from latest commit are:")
        for item in repo.index.diff(None):
            log.info(f"{item.a_path}")
        log.info("----")

        log.info(f"Work directory is {self.work_dir}")
        log.info("Running with configuration:\n" + OmegaConf.to_yaml(self.cfg))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            log.info("Found {} CUDA devices.".format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log.info(
                    "{} \t Memory: {:.2f}GB".format(
                        props.name, props.total_memory / (1024 ** 3)
                    )
                )
            torch.backends.cudnn.benchmark = True
        else:
            log.info("WARNING: Using device {}".format(device))

        self.device = device
        self.use_gpu = device.type == "cuda"

        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        self.main()

    def initialize(self):

        # Populates self.target0, self.target1, self.dim
        self.load_targets()

        self.f_icnn = ICNN(
            input_dim=self.dim,
            hidden_size=self.cfg.d_model,
            num_layers=self.cfg.nhidden,
        ).to(self.device)
        self.g_icnn = ICNN(
            input_dim=self.dim,
            hidden_size=self.cfg.d_model,
            num_layers=self.cfg.nhidden,
        ).to(self.device)

        self.f_icnn.project()

        log.info(self.f_icnn)
        log.info(self.g_icnn)

        self.optimizer_f = torch.optim.Adam(
            self.f_icnn.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9)
        )
        self.optimizer_g = torch.optim.Adam(
            self.g_icnn.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9)
        )
        self.loss_g_meter = utils.RunningAverageMeter(0.99)
        self.loss_f_meter = utils.RunningAverageMeter(0.99)

    def main(self):
        self.initialize()

        self.lim = 2

        start_time = time.time()
        prev_itr = self.iter - 1
        while self.iter <= self.cfg.num_iterations:

            for _ in range(self.cfg.inner_loop_iterations):

                # Train g
                x1 = self.target1.sample(self.cfg.batch_size)

                x1.requires_grad_(True)
                g = self.g_icnn(x1).sum()
                grad_g = torch.autograd.grad(g, x1, create_graph=True)[0]

                f_grad_gy = self.f_icnn(grad_g).mean()
                y_dot_grad_gy = (x1 * grad_g).sum(1).mean()
                reg = self.g_icnn.cvx_regularization()
                loss_g = f_grad_gy - y_dot_grad_gy + self.cfg.reg_coef * reg

                self.loss_g_meter.update(loss_g.item())
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

            # Train f
            x0 = self.target0.sample(self.cfg.batch_size)
            x1 = self.target1.sample(self.cfg.batch_size)

            x1.requires_grad_(True)
            g = self.g_icnn(x1).sum()
            grad_g = torch.autograd.grad(g, x1, create_graph=True)[0]
            f_grad_gy = self.f_icnn(grad_g).mean()
            fx = self.f_icnn(x0).mean()

            loss_f = fx - f_grad_gy

            self.loss_f_meter.update(loss_f.item())

            self.optimizer_f.zero_grad()
            loss_f.backward()
            self.optimizer_f.step()
            self.f_icnn.project()

            if self.iter % self.cfg.logfreq == 0:
                time_per_itr = (time.time() - start_time) / (self.iter - prev_itr)
                prev_itr = self.iter
                log.info(
                    f"Iter {self.iter}"
                    f" | Time {time_per_itr:.4f}"
                    f" | Loss-G {self.loss_g_meter.val:.4f}({self.loss_g_meter.avg:.4f})"
                    f" | Loss-F {self.loss_f_meter.val:.4f}({self.loss_f_meter.avg:.4f})"
                )
                self.save()
                start_time = time.time()

            if self.iter % self.cfg.evalfreq == 0:
                self.evaluate(5000)

            self.iter += 1

    @torch.no_grad()
    def simulate(self, num_samples):
        x1 = self.target1.sample(num_samples).to(self.device)

        with torch.enable_grad():
            x1 = x1.requires_grad_(True)
            g = self.g_icnn(x1).sum()
            x0 = torch.autograd.grad(g, x1)[0]
        return x0, x1

    @torch.no_grad()
    def evaluate(self, num_samples):
        x0, x1 = self.simulate(num_samples)
        dist = x0 - x1
        cost = torch.sum(dist * dist, dim=-1).mean()

        log.info(f"Iter {self.iter} | Estimated W2 {cost:.8f}")

    def load_targets(self):

        self.dim = self.cfg.dim

        if self.cfg.dim == 2:
            self.target0 = load_2dtarget(self.cfg.target0)
            self.target1 = load_2dtarget(self.cfg.target1)
        else:
            self.dim = self.cfg.dim

            rng = np.random.RandomState(self.dim)
            centers0 = rng.randn(3, self.dim) * 0.3
            centers1 = rng.randn(6, self.dim) * 0.3
            self.target0 = GaussianMM(centers0, std=1 / np.sqrt(self.dim))
            self.target1 = GaussianMM(centers1, std=1 / np.sqrt(self.dim))

        self.target0.float().to(self.device)
        self.target1.float().to(self.device)

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)


# Import like this for pickling
from main_ot_icnn import Workspace as W


@hydra.main(config_path="configs", config_name="ot_icnn")
def main(cfg):
    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        log.info(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
            workspace.cfg = cfg
    else:
        workspace = W(cfg)

    try:
        workspace.run()
    except Exception as e:
        log.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
