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
from functorch import vmap
from torchdiffeq import odeint

from dist2d import load_2dtarget
from dist2d import GaussianMM
from divfree import build_divfree_vector_field
from model import NeuralConservationLaw
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
        self.x0_fixed = self.target0.sample(self.cfg.num_test_samples).to(self.device)
        self.x1_fixed = self.target1.sample(self.cfg.num_test_samples).to(self.device)

        if not hasattr(self, "module"):
            self.module = NeuralConservationLaw(
                self.dim + 1,
                d_model=self.cfg.d_model,
                num_hidden_layers=self.cfg.nhidden,
                n_mixtures=self.cfg.nmix,
                actfn=self.cfg.actfn,
            ).to(self.device)
            u_fn, params, _ = build_divfree_vector_field(self.module)
            self.params = params
            self.optimizer = torch.optim.Adam(self.params, lr=self.cfg.lr)
            self.loss_meter = utils.RunningAverageMeter(0.99)
        else:
            u_fn, _, _ = build_divfree_vector_field(self.module)

        log.info(self.module)

        return u_fn

    def main(self):
        u_fn = self.initialize()
        u_fn = vmap(u_fn, in_dims=(None, 0))

        self.lim = 2

        start_time = time.time()
        prev_itr = self.iter - 1
        while self.iter <= self.cfg.num_iterations:

            # Sample from target distributions.
            x0 = self.target0.sample(self.cfg.batch_size)
            x1 = self.target1.sample(self.cfg.batch_size)

            # Include random samples.
            xu = (
                torch.rand(self.cfg.batch_size, self.cfg.dim).to(x0) * 2 * self.lim
                - self.lim
            )
            x0 = torch.cat([x0, xu], dim=0)
            x1 = torch.cat([x1, xu], dim=0)

            y0 = torch.cat([torch.zeros(x0.shape[0], 1).to(x0), x0], dim=1)
            y1 = torch.cat([torch.ones(x1.shape[0], 1).to(x1), x1], dim=1)
            y = torch.cat([y0, y1], dim=0)

            rho = u_fn(self.params, y)[..., 0]

            if torch.any(rho < 0):
                min_rho = torch.min(rho).item()
                log.info(f"WARNING: rho < 0: {min_rho}")

            rho = rho.clamp(min=1e-8)
            rho0, rho1 = torch.split(rho, rho.shape[0] // 2, dim=0)

            # Fit rho(t=0).
            p0 = torch.exp(self.target0.logprob(x0))
            loss0 = (p0 - rho0).abs().mean()

            # Fit rho(t=1).
            p1 = torch.exp(self.target1.logprob(x1))
            loss1 = (p1 - rho1).abs().mean()

            # Optimize for optimal transport.
            x0 = self.target0.sample(self.cfg.batch_size)
            x1 = self.target1.sample(self.cfg.batch_size)
            p0 = torch.exp(self.target0.logprob(x0))
            p1 = torch.exp(self.target1.logprob(x1))
            pu = torch.tensor(1 / (2 * self.lim) ** self.dim).expand(xu.shape[0]).to(p0)

            x = torch.cat([x0, x1, xu], dim=0)
            y = torch.cat([torch.rand(x.shape[0], 1).to(x), x], dim=1)
            ps = torch.cat([p0, p1, pu], dim=0)

            u = u_fn(self.params, y)
            v = u[..., 1:] / u[..., 0:1].clamp(min=1e-8)
            reg = torch.sum(v * v, dim=-1) * u[..., 0].clamp(min=1e-8) / ps
            loss_v = reg.mean()

            Lcoef = self.cfg.lambd_coef
            loss = Lcoef * (loss0 + loss1) + loss_v
            self.loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 1)

            if ~torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"Gradient norm is {grad_norm} while loss is {loss.item()}"
                )

            self.optimizer.step()

            if self.iter % self.cfg.logfreq == 0:
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

            if self.iter % self.cfg.evalfreq == 0:
                self.evaluate(u_fn)

            if self.iter % self.cfg.vizfreq == 0:
                self.visualize(u_fn, plot_samples=True)

            self.iter += 1

    @torch.no_grad()
    def simulate(self, u_fn):
        # Sample from p0
        x0 = self.x0_fixed

        def v_fn(t, x):
            t = torch.ones(x.shape[0], 1).to(x) * t
            y = torch.cat([t, x], dim=1)
            u = u_fn(self.params, y)
            v = u[:, 1:] / u[:, 0:1].clamp(min=1e-8)
            v = torch.where(u[:, 0:1] <= 1e-8, torch.zeros_like(v), v)
            return v

        options = {}
        if self.iter < 100:
            options["min_step"] = 0.01

        # Transform through learned vector field.
        xs = odeint(
            v_fn,
            x0,
            t=torch.linspace(0, 1, 5).to(self.device),
            method="dopri5",
            atol=1e-5,
            rtol=1e-5,
            options=options,
        )
        return xs

    @torch.no_grad()
    def evaluate(self, u_fn):
        samples = self.simulate(u_fn)
        dist = samples[0] - samples[-1]
        cost = torch.sum(dist * dist, dim=-1).mean()

        x0 = self.x0_fixed
        logp0 = self.target0.logprob(x0)
        x1 = self.x1_fixed
        logp1 = self.target1.logprob(x1)

        y0 = torch.cat([torch.zeros(x0.shape[0], 1).to(x0), x0], dim=1)
        y1 = torch.cat([torch.ones(x1.shape[0], 1).to(x1), x1], dim=1)
        y = torch.cat([y0, y1], dim=0)

        rho = u_fn(self.params, y)[..., 0]
        rho0, rho1 = torch.split(rho, rho.shape[0] // 2, dim=0)

        logrho0 = torch.log(torch.clamp(rho0, min=1e-8))
        logrho1 = torch.log(torch.clamp(rho1, min=1e-8))

        kl0 = (logp0 - logrho0).mean().item()
        kl1 = (logp1 - logrho1).mean().item()

        log.info(
            f"Iter {self.iter} | Estimated W2 {cost:.8f}"
            f" | KL (t=0): {kl0:.8f} | KL (t=1): {kl1:.8f}"
        )

    @torch.no_grad()
    def visualize(self, u_fn, plot_samples):

        os.makedirs("figs", exist_ok=True)

        # Sample from the model
        if plot_samples:
            samples = self.simulate(u_fn)
            samples = samples.cpu().numpy()

            data0 = self.target0.sample(1000).cpu().numpy()
            data1 = self.target1.sample(1000).cpu().numpy()

            samples = samples[:, :, :2]
            data0 = data0[:, :2].reshape(-1, 2)
            data1 = data1[:, :2].reshape(-1, 2)

            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
            axs[0].scatter(data0[:, 0], data0[:, 1], s=1, marker="x", c="C0")
            axs[-1].scatter(data1[:, 0], data1[:, 1], s=1, marker="x", c="C0")
            for i in range(5):
                axs[i].scatter(
                    samples[i, :, 0], samples[i, :, 1], s=1, marker="o", c="C1"
                )
                axs[i].get_xaxis().set_ticks([])
                axs[i].get_yaxis().set_ticks([])
                if self.dim == 2:
                    axs[i].set_ylim([-2, 2])
                    axs[i].set_xlim([-2, 2])
                else:
                    axs[i].set_ylim([-2, 2])
                    axs[i].set_xlim([-2, 2])
            fig.tight_layout()
            plt.savefig(f"figs/samples-{self.iter:06d}.png")
            plt.close()

        if self.dim != 2:
            return

        def density_fn(x, t):
            y = torch.cat([torch.ones(x.shape[0], 1).to(x) * t, x], dim=1)
            rho = u_fn(self.params, y)[..., 0]
            return rho

        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
        plt_density(partial(density_fn, t=0.00), axs[0], device=self.device)
        plt_density(partial(density_fn, t=0.25), axs[1], device=self.device)
        plt_density(partial(density_fn, t=0.50), axs[2], device=self.device)
        plt_density(partial(density_fn, t=0.75), axs[3], device=self.device)
        plt_density(partial(density_fn, t=1.00), axs[4], device=self.device)
        fig.tight_layout()
        plt.savefig(f"figs/rho-{self.iter:06d}.png")
        plt.close()

        def vecfield_fn(x, t):
            y = torch.cat([torch.ones(x.shape[0], 1).to(x) * t, x], dim=1)
            u = u_fn(self.params, y)
            v = u[..., 1:] / u[..., 0:1].clamp(min=1e-8)
            return v

        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
        plt_vecfield(partial(vecfield_fn, t=0.00), axs[0], device=self.device)
        plt_vecfield(partial(vecfield_fn, t=0.25), axs[1], device=self.device)
        plt_vecfield(partial(vecfield_fn, t=0.50), axs[2], device=self.device)
        plt_vecfield(partial(vecfield_fn, t=0.75), axs[3], device=self.device)
        plt_vecfield(partial(vecfield_fn, t=1.00), axs[4], device=self.device)
        fig.tight_layout()
        plt.savefig(f"figs/vecfield-{self.iter:06d}.png")
        plt.close()

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
from main_ot import Workspace as W


@hydra.main(version_base=None, config_path="configs", config_name="ot")
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
