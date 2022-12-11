"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax.numpy as jnp
from jax import jacfwd
import pickle

#divergence operator -- defined on tensors of all orders
#uses row convention on 2-tensors (matrices)
def div(F):
    B = jacfwd(F)
    return lambda x: jnp.trace(B(x),axis1=-2,axis2=-1)

#analog of curl by taking norm of Df - Df^T
def curl(F):
    B = jacfwd(F)
    return lambda x: jnp.sum(jnp.power(B(x) - B(x).T,2))

def saveState(params, opt_st, stats,path):
    with open(path + "_model",'wb') as f:
        pickle.dump(params,f)
    with open(path + "_stats",'wb') as f:
        pickle.dump(stats,f)
    # with open(path + "_opt",'wb') as f:
    #     pickle.dump(opt_st,f)
        
def loadState(path):
    with open(path + "_model",'rb') as f:
        params = pickle.load(f)
    # with open(path + "_opt",'rb') as f:
    #     opt_st = pickle.load(f)
    return params

#define periodic embedding for fixed k
def periodic(x,k=-1/2):
    c = jnp.pi*2**(k+1)
    return jnp.concatenate([jnp.cos(c*x),jnp.sin(c*x)])
