"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models import NCLImplicit,MLP,NCL,DivFree,DivFreeImplicit,MixtureScore,DivFreeSparse
import numpy as np
import jax.random as random
import jax
import jax.numpy as jnp
from jax import grad,vmap
from losses import HelmholtzLoss
from hh_experiment_setup import runHelmholtzExperiment,runHelmholtzExperiment2
import optax 
from utils import div

# Runs the Helmholtz experiment with a div-free net

#define hyperparams for div free model
beta = 25
act = jnp.sin#lambda x: jax.nn.softplus(x*beta)/beta
#act = lambda x: jax.nn.softplus(x*beta)**2/(beta*beta*2)
layers = 3
width = 128

seed = 24#np.random.randint(2**32)
key =  random.PRNGKey(seed)
print("Random initial seed:", seed)
dim = 10

#define periodic embedding for fixed k
def periodic(x,k):
    c = jnp.pi*2**(k+1)
    return jnp.concatenate([cos(c*x),sin(c*x)])

#sets up the div_free approximator
x = random.normal(key,shape=(dim+1,))
mlp = MLP(depth=layers,width=width,act=act,out_dim=dim*(dim-1)//2,std=1,bias=True)
key, mkey = random.split(key,2)
params = mlp.init(mkey,x[1:])
scale = 1.4
params = jax.tree_map(lambda x: x*scale, params)
params = params.unfreeze()['params']

func_mlp = lambda x,params: mlp.apply({'params':params}, x)

# def div_free(x,params):
#     ncl = DivFree(lambda y,params: func_mlp(jnp.array([x[0],*y]),params))
#     return 0.1*ncl(x[1:],params)
div_free = lambda x,params: DivFree(func_mlp)(x[1:],params)
print("Sample NCL output:", div_free(x,params))


sched = optax.piecewise_constant_schedule(init_value=1e-2,
                                    boundaries_and_scales={300:1e-2,
                                                           4000:1e-2}
                                   )
opt = optax.adam(learning_rate=1e-2,b1=0.9,b2=0.99)
#opt = optax.sgd(learning_rate=1e-3,momentum=0.9)
runHelmholtzExperiment(params, 
    key, 
    model=div_free,
    apx=str(seed)+"matrix_ncl",
    opt=opt,
    eps=10000,
    dim=dim,
    m_type='curl',
                      )




