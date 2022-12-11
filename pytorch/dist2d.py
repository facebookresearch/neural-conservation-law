"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import numpy as np
import sklearn.datasets

import torch
import torch.nn as nn


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


class GaussianMM(nn.Module):
    def __init__(self, centers, std):
        super().__init__()
        self.register_buffer("centers", torch.tensor(centers))
        self.register_buffer("logstd", torch.tensor(std).log())
        self.K = centers.shape[0]

    def logprob(self, x):
        """Computes the log probability."""
        logprobs = normal_logprob(
            x.unsqueeze(1), self.centers.unsqueeze(0), self.logstd
        )
        logprobs = torch.sum(logprobs, dim=2)
        return torch.logsumexp(logprobs, dim=1) - math.log(self.K)

    def sample(self, n_samples):
        idx = torch.randint(self.K, (n_samples,)).to(self.centers.device)
        mean = self.centers[idx]
        return torch.randn_like(mean) * torch.exp(self.logstd) + mean


def load_2dtarget(target):
    if target == "circles":
        centers = sklearn.datasets.make_circles(
            1000, shuffle=False, random_state=0, factor=0.5
        )[0]
        return GaussianMM(centers * 1.5, std=0.16)
    elif target == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = np.array(centers) * scale
        return GaussianMM(centers / 2.828, 0.5 / 2.828)
    elif target == "pinwheel":
        rng = np.random.RandomState(0)
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = 1000
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes * num_per_class, 2) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        centers = rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        return GaussianMM(centers, 0.1)
    elif target == "swissroll":
        centers = sklearn.datasets.make_swiss_roll(
            n_samples=5000, noise=0, random_state=0
        )[0]
        centers = centers[:, [0, 2]] / 10
        return GaussianMM(centers, 0.1)
    elif target == "moons":
        centers = sklearn.datasets.make_moons(n_samples=5000, noise=0, random_state=0)[
            0
        ]
        centers = centers.astype("float32")
        centers = (centers * 2 + np.array([-1, -0.2])) / 2
        return GaussianMM(centers, 0.1)
    elif target == "2spirals":
        rng = np.random.RandomState(0)
        n = np.sqrt(rng.rand(5000 // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(5000 // 2, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(5000 // 2, 1) * 0.5
        centers = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        return GaussianMM(centers / 2, 0.1)
    elif target == "gaussian":
        return GaussianMM(torch.zeros(1, 2), std=1.0)
    else:
        raise ValueError(f"Unknown target {target}")
