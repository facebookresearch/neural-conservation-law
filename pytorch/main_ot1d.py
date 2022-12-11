"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from omegaconf import OmegaConf
import hydra

import git
import time
import logging
import os
import random
import numpy as np
import pickle as pkl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from functorch import vmap
from torchdiffeq import odeint

from divfree import build_divfree_vector_field
from distributions import normal_logprob, mixture_normal_logprob, mixture_normal_sample
from model import NeuralConservationLaw
import utils

log = logging.getLogger(__name__)


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

        if not hasattr(self, "module"):
            self.module = NeuralConservationLaw(2, d_model=128).to(self.device)
            u_fn, params, _ = build_divfree_vector_field(self.module)
            self.params = params
            self.optimizer = torch.optim.Adam(self.params, lr=0.005)
            self.loss_meter = utils.RunningAverageMeter(0.99)
        else:
            u_fn, _, _ = build_divfree_vector_field(self.module)

        return u_fn

    def main(self):
        u_fn = self.initialize()
        u_fn = vmap(u_fn, in_dims=(None, 0))

        if self.cfg.target == 0:

            def target_logprob(x):
                means = torch.tensor([3.0]).reshape(1, 1).expand(x.shape[0], 1).to(x)
                logstds = (
                    torch.tensor([0.4]).log().reshape(1, 1).expand(x.shape[0], 1).to(x)
                )
                logits = torch.zeros(1, 1).to(x)
                return mixture_normal_logprob(x, means, logstds, logits)

        elif self.cfg.target == 1:

            def target_logprob(x):
                means = (
                    torch.tensor([-1.5, 0.0, 1.5])
                    .reshape(1, 3)
                    .expand(x.shape[0], 3)
                    .to(x)
                )
                logstds = (
                    torch.tensor([0.2, 0.3, 0.4])
                    .log()
                    .reshape(1, 3)
                    .expand(x.shape[0], 3)
                    .to(x)
                )
                logits = torch.zeros(1, 3).to(x)
                return mixture_normal_logprob(x, means, logstds, logits)

        else:
            raise ValueError(f"Unknown target {self.cfg.target}")

        start_time = time.time()
        prev_itr = self.iter - 1
        while self.iter < self.cfg.num_iterations:

            x = torch.linspace(0, 1, self.cfg.batch_size)
            x = x + torch.rand(self.cfg.batch_size) / (self.cfg.batch_size - 1)
            x = x * 12 - 6
            x = x.to(self.device)

            y0 = torch.stack([torch.zeros_like(x), x], dim=-1)
            y1 = torch.stack([torch.ones_like(x), x], dim=-1)
            y = torch.cat([y0, y1], dim=0)

            rho = u_fn(self.params, y)[..., 0]
            rho0, rho1 = torch.split(rho, rho.shape[0] // 2, dim=0)

            # Fit rho(t=0) to be a standard Normal.
            p0 = torch.exp(normal_logprob(x, 0.0, 0.0))
            loss0 = (p0 - rho0).abs().mean()

            # Fit rho(t=1) to the target distribution.
            p1 = torch.exp(target_logprob(x))
            loss1 = (p1 - rho1).abs().mean()

            # Satisfy optimal transport.
            y = torch.stack([torch.rand_like(x), x], dim=-1)
            u = u_fn(self.params, y)
            reg = torch.linalg.norm(u[..., 1:], dim=-1) / u[..., 0]
            loss_v = reg.mean()

            # Boundary conditions?
            y = torch.tensor([[0.0, -10.0], [0.0, 10.0], [1.0, -10.0], [1.0, 10.0]]).to(
                self.device
            )
            rho0 = u_fn(self.params, y)[..., 0]
            loss_b = rho0.abs().mean()

            Lcoef = min(self.iter / self.cfg.num_iterations, 1) * self.cfg.ot_coef
            loss = loss0 + loss_b + loss1 + loss_v * Lcoef
            self.loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 1e20)
            self.optimizer.step()

            if self.iter % 10 == 0:
                time_per_itr = (time.time() - start_time) / (self.iter - prev_itr)
                prev_itr = self.iter
                log.info(
                    f"Iter {self.iter}"
                    f" | Time {time_per_itr:.4f}"
                    f" | Loss {self.loss_meter.val:.4f}({self.loss_meter.avg:.4f})"
                    f" | GradNorm {grad_norm:.4f}"
                    f" | LCoef {Lcoef:.6f}"
                )
                self.save()
                start_time = time.time()

            if self.iter % 1000 == 0:
                self.visualize(u_fn, target_logprob, plot_samples=self.iter > 2000)

            self.iter += 1

    @torch.no_grad()
    def simulate(self, u_fn, num_samples):
        # Sample from p0
        x0 = torch.randn(num_samples).to(self.device)

        def v_fn(t, x):
            t = torch.ones_like(x) * t
            y = torch.stack([t, x], dim=-1)
            u = u_fn(self.params, y)
            v = u[:, 1] / u[:, 0]
            return v

        # Transform through learned vector field.
        xs = odeint(
            v_fn,
            x0,
            t=torch.linspace(0, 1, 5).to(self.device),
            method="rk4",
            options={"step_size": 0.01},
        )
        return xs

    @torch.no_grad()
    def visualize(self, u_fn, target_logprob, plot_samples):
        x = torch.linspace(-5, 5, 1000)
        x = x.to(self.device)

        y = torch.cat(
            [
                torch.stack([torch.ones_like(x) * 0, x], dim=-1),
                torch.stack([torch.ones_like(x) * 1 / 4, x], dim=-1),
                torch.stack([torch.ones_like(x) * 2 / 4, x], dim=-1),
                torch.stack([torch.ones_like(x) * 3 / 4, x], dim=-1),
                torch.stack([torch.ones_like(x) * 1, x], dim=-1),
            ],
            dim=0,
        )
        rho = u_fn(self.params, y)[..., 0].reshape(5, -1)

        p0 = torch.exp(normal_logprob(x, 0.0, 0.0))
        p1 = torch.exp(target_logprob(x))

        # Sample from the model
        if plot_samples:
            samples = self.simulate(u_fn, num_samples=1000)
            samples = samples.cpu().numpy()
        else:
            samples = None

        x = x.cpu().numpy()
        rho = rho.cpu().numpy()
        p0 = p0.cpu().numpy()
        p1 = p1.cpu().numpy()

        fig, axs = plt.subplots(
            figsize=(4 * 5, 4), nrows=1, ncols=5, sharex=True, sharey=True
        )
        axs[0].plot(x, rho[0], label=r"$\rho_0$", color="C0")
        axs[0].plot(x, p0, "k--", label=r"$p_0$", alpha=0.5)
        if samples is not None:
            axs[0].hist(
                samples[0], bins=100, range=(-5, 5), density=True, color="C2", alpha=0.3
            )
        axs[0].legend()
        axs[1].plot(x, rho[1], label=r"$\rho_{t=1/4}$")
        if samples is not None:
            axs[1].hist(
                samples[1], bins=100, range=(-5, 5), density=True, color="C1", alpha=0.3
            )
        axs[1].legend()
        axs[2].plot(x, rho[2], label=r"$\rho_{t=2/4}$")
        if samples is not None:
            axs[2].hist(
                samples[2], bins=100, range=(-5, 5), density=True, color="C1", alpha=0.3
            )
        axs[2].legend()
        axs[3].plot(x, rho[3], label=r"$\rho_{t=3/4}$")
        if samples is not None:
            axs[3].hist(
                samples[3], bins=100, range=(-5, 5), density=True, color="C1", alpha=0.3
            )
        axs[3].legend()
        axs[4].plot(x, rho[4], label=r"$\rho_1$", color="C0")
        axs[4].plot(x, p1, "k--", label=r"$p_1$", alpha=0.5)
        if samples is not None:
            axs[4].hist(
                samples[4], bins=100, range=(-5, 5), density=True, color="C1", alpha=0.3
            )
        axs[4].legend()
        fig.tight_layout()
        fig.savefig("rho0.png")
        plt.close()

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)


# Import like this for pickling
from main_ot1d import Workspace as W


@hydra.main(config_path="configs", config_name="ot1d")
def main(cfg):
    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        log.info(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    try:
        workspace.run()
    except Exception as e:
        log.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
