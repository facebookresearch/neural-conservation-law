"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import click
import numpy as np
import ot

from dist2d import load_2dtarget


@click.group()
def cli():
    pass


@cli.command()
@click.option("--target0", type=click.STRING, default="pinwheel")
@click.option("--target1", type=click.STRING, default="8gaussians")
@click.option("--nsamples", type=click.INT, default=10000)
def solve_ot(
    target0: str, target1: str, nsamples: int,
):
    target0 = load_2dtarget(target0)
    target1 = load_2dtarget(target1)

    samples0 = target0.sample(nsamples).double().cpu().numpy()
    samples1 = target1.sample(nsamples).double().cpu().numpy()

    prod0 = np.ones(nsamples) / nsamples
    prod1 = np.ones(nsamples) / nsamples

    M = ot.dist(samples0, samples1)
    wdist = ot.emd2(prod0, prod1, M, numItermax=1000000)

    print(f"cost {wdist}")


if __name__ == "__main__":
    cli()

