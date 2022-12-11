"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logprob(z, mean, logstd):
    mean = mean + torch.tensor(0.0)
    logstd = logstd + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-logstd)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logstd + c)


def mixture_normal_logprob(z, means, logstds, logits):
    z = z.unsqueeze(-1)
    logpi = F.log_softmax(logits, -1)
    logp_mixtures = normal_logprob(z, means, logstds)
    logp = torch.logsumexp(logpi + logp_mixtures, dim=-1)
    return logp


def mixture_normal_sample(means, logstds, logits):
    K = logits.shape[-1]
    logpi = F.log_softmax(logits, -1)
    mask = F.one_hot(Categorical(logits=logpi).sample(), K).bool()
    z = torch.randn_like(means) * torch.exp(logstds) + means
    z = torch.masked_select(z, mask).reshape(means.shape[:-1])
    return z


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    means = torch.tensor([-1.5, 0., 1.5]).reshape(1, 3)
    logstds = torch.tensor([0.2, 0.3, 0.4]).log().reshape(1, 3)
    logits = torch.tensor([0., 0., 0.]).reshape(1, 3)

    x = torch.linspace(-5, 5, 200)
    logps = mixture_normal_logprob(x, means, logstds, logits)

    plt.plot(x, logps.exp())
    plt.savefig("mixture_logps.png")
