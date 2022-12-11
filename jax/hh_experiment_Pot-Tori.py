"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models import NCLImplicit,MLP,NCL,DivFree,DivFreeImplicit,MixtureScore
import numpy as np
import jax.random as random
import jax
import jax.numpy as jnp
from jax import grad,vmap
from losses import HelmholtzLoss
from hh_experiment_setup import runHelmholtzExperiment3
import optax 
from utils import div, periodic

# Runs the Helmholtz experiment with the gradient of a scalar network

#define hyperparams for u,rho,p
beta = 25
act = jnp.sin#lambda x: jax.nn.softplus(x*beta)/beta
#act = lambda x: jax.nn.softplus(x*beta)**2/(beta*beta*2)
layers = 3
width = 128

seed = 28#np.random.randint(2**32)
key =  random.PRNGKey(seed)
print("Random initial seed:", seed)
dim = 10

#sets up the potential approximator
x = random.normal(key,shape=(dim+1,))
mlp = MLP(depth=layers,width=width,act=act,out_dim=1,std=1,bias=False)
params = mlp.init(key,periodic(x[1:]))
scale = 1.5
params = jax.tree_map(lambda x: x*scale, params)
params = params.unfreeze()['params']

func_mlp = lambda x,params: mlp.apply({'params':params}, periodic(x[1:]))[0]
div_free = lambda x,params: x[1:] - grad(lambda y: func_mlp(jnp.array([x[0],*y]), params))(x[1:])
print("Sample NCL output:", div_free(x,params))


sched = optax.piecewise_constant_schedule(init_value=1e-2,
                                    boundaries_and_scales={300:1e-2,
                                                           4000:1e-2}
                                   )
opt = optax.adam(learning_rate=2e-2,b1=0.9,b2=0.99)
#opt = optax.sgd(learning_rate=1e-3,momentum=0.6)
runHelmholtzExperiment3(params, 
    key, 
    model=div_free,
    apx=str(seed)+"matrix_ncl",
    opt=opt,
    eps=1000,
    dim=dim,
    m_type='l2')




