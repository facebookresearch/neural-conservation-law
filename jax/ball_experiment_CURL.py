"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models import NCLImplicit,MLP
import numpy as np
import jax.random as random
import jax
import jax.numpy as jnp
from pde import PDE
from losses import Loss
from ball_experiment_setup import runBallExperiment
import optax 

# Runs the Ball experiment with a Div-Free PINN model

#define hyperparams for u,rho,p
beta = 25
act = lambda x: jax.nn.softplus(x*beta)/beta
#act = lambda x: jax.nn.softplus(x*beta)**2/(beta*beta*2)
layers = 4
width = 128

seed = np.random.randint(2**32)
key =  random.PRNGKey(seed)
print("Random initial seed:", seed)
x = random.normal(key,shape=(4,))
mlp = MLP(depth=layers,width=width,act=act,out_dim=5,std=1)
params = mlp.init(key,x)

scale = 8e-1
params = jax.tree_map(lambda x: x*scale, params)
params = params.unfreeze()['params']

func_mlp = lambda x,params: mlp.apply({'params':params}, x)

#this is a bad hack
def curl_pinn(x,params):
    ncl = NCLImplicit(lambda y,params: func_mlp(jnp.array([x[0],*y]),params)[:-1])
    u_x = func_mlp(x,params)
    return jnp.array([u_x[-1], *ncl(x[1:],params)])

print("Sample CURL output:", curl_pinn(x,params))

#convenience for plotting, only curl is passed to train/loss module
u = lambda x,params: curl_pinn(x,params)[1:-1]
rho = lambda x,params: curl_pinn(x,params)[0]

pde = PDE()
pde.setNormal(lambda y: y[1:])
loss = Loss(curl_pinn)
loss.addTermDom(pde.mom,'mom')
loss.addTermDom(pde.cont,'cont')
loss.addTermInit(pde.init,'init')
loss.addTermBd(pde.bd, 'fs')

gamma = {
    'mom':2e-1,
    'cont':3e0,
    'fs':1e-1,
    'init':3e1
}
loss.setGamma(gamma)

sched = optax.piecewise_constant_schedule(init_value=3e-3,
                                    boundaries_and_scales={300:1e-2}
                                   )

runBallExperiment(params=params, 
                  key=key,
                  pde=pde,
                  loss=loss,
                  u=u,
                  rho=rho,
                  sched=sched,
                  apx=str(seed)+"curl",
                  
                    )




