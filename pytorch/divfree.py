"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from functorch import make_functional
from functorch import vmap
from functorch import jacrev


def div(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def build_divfree_vector_field(module):
    """Returns an unbatched vector field, i.e. assumes input is a 1D tensor."""

    F_fn, params = make_functional(module)

    J_fn = jacrev(F_fn, argnums=1)

    def A_fn(params, x):
        J = J_fn(params, x)
        A = J - J.T
        return A

    def A_flat_fn(params, x):
        A = A_fn(params, x)
        A_flat = A.reshape(-1)
        return A_flat

    def ddF(params, x):
        D = x.nelement()
        dA_flat = jacrev(A_flat_fn, argnums=1)(params, x)
        Jac_all = dA_flat.reshape(D, D, D)
        ddF = vmap(torch.trace)(Jac_all)
        return ddF

    return ddF, params, A_fn


if __name__ == "__main__":

    torch.manual_seed(0)

    bsz = 10
    ndim = 5

    module = nn.Sequential(
        nn.Linear(ndim, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, ndim),
    )

    u_fn, params, A_fn = build_divfree_vector_field(module)

    x = torch.randn(bsz, ndim)
    A = vmap(A_fn, in_dims=(None, 0))(params, x)
    print("A should be antisymmetric:")
    print(A.shape)
    print(A)

    u = vmap(u_fn, in_dims=(None, 0))(params, x)
    print("vector field u:")
    print(u.shape)
    print(u)

    div_u = div(lambda x: u_fn(params, x))
    d = vmap(div_u)(x)
    print("Divergence(u) should be zero:")
    print(d.shape)
    print(d)

