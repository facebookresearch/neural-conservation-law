"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models import MLP
import numpy as np
import jax.numpy as jnp
import jax.random as random
import jax
from pde import PDE
from losses import Loss
from tori_experiment_setup import runToriExperiment,embd
import optax
import argparse
#runs the TORI experiment with a standard PINN

parser = argparse.ArgumentParser('Tori Experiment (PINN)')
parser.add_argument('--seed', type=int,default=0, help='seed for rng')

# Runs the Tori experiment with a NCL model
args = parser.parse_args()

#define hyperparams for u,rho,p
beta = 1
act = jnp.sin#lambda x: jax.nn.softplus(x*beta)/beta
#act = lambda x: jax.nn.softplus(x*beta)**2/(beta*6)
#choose spectral frequencies for periodic encodings 
freqs = []
layers = 8
width = 512

#pinn mlp outputs [rho,u,p] (u=u_x,u_y)
pinn_mlp = MLP(depth=layers,width=width,act=act,out_dim=4,std=1,bias=True)

if not args.seed:
    seed = np.random.randint(2**32)
else:
    seed = args.seed
key =  random.PRNGKey(seed)
print("Random initial seed:", seed)
x = random.normal(key,shape=(3,))
params = pinn_mlp.init(key,embd(x,freqs)).unfreeze()['params']

#define curried mlp (flax makes this necesary?)
pinn = lambda x,params: pinn_mlp.apply({'params':params}, embd(x,freqs))

#convenience for plotting, only pinn is passed to train/loss module
u = lambda x,params: pinn(x,params)[1:3]
rho = lambda x,params: pinn(x,params)[0]

pde = PDE()
loss = Loss(pinn)
loss.addTermDom(pde.mom,'mom')
loss.addTermDom(pde.cont,'cont')
loss.addTermDom(pde.incp,'incp')
loss.addTermInit(pde.init,'init')

gamma = {
    'mom':1e-2,
    'cont':1e3,
    'incp':1e-1,
    'init':3e1
}
loss.setGamma(gamma)

sched = optax.piecewise_constant_schedule(init_value=1e-4,
                                    boundaries_and_scales={300:0.5,}
                                   )


runToriExperiment(params=params, 
                  key=key,
                  pde=pde,
                  loss=loss,
                  sched=sched,
                  u=u,
                  rho=rho,
                  load_path='',
                  apx=str(seed)+"pinn")




