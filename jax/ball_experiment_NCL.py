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
from pde import PDEDivForm
from losses import Loss
from ball_experiment_setup import runBallExperiment
import optax 

# Runs the Ball experiment with a NCL model

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

#ncl outputs [rho,rho u, p] u = (u_x,u_y,u_z)
ncl = NCLImplicit(func_mlp)

print("Sample NCL output:", ncl(x,params))

#convenience for plotting, only ncl is passed to train/loss module
u = lambda x,params: ncl(x,params)[1:3]/ncl(x,params)[0]
rho = lambda x,params: ncl(x,params)[0]

pde = PDEDivForm()
pde.setNormal(lambda y: y[1:])
loss = Loss(ncl)
loss.addTermDom(pde.mom,'mom')
loss.addTermDom(pde.incp,'incp')
loss.addTermInit(pde.init,'init')
loss.addTermBd(pde.bdry, 'fs')

gamma = {
    'mom':2e-1,
    'incp':1e-1,
    'fs':1e-1,
    'init':3e1
}
loss.setGamma(gamma)

sched = optax.piecewise_constant_schedule(init_value=1e-3,
                                    boundaries_and_scales={300:5e-3,
                                                           50000:3e-3}
                                   )

runBallExperiment(params=params, 
                  key=key,
                  pde=pde,
                  loss=loss,
                  u=u,
                  rho=rho,
                  sched=sched,
                  apx=str(seed)+"ncl",
                    )




