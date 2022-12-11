"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models import NCL,MLP, NCL_sparse
import numpy as np
import jax.random as random
import jax
import jax.numpy as jnp
from pde import PDEDivForm
from losses import Loss
from tori_experiment_setup import runToriExperiment,embd
import optax 
import argparse


parser = argparse.ArgumentParser('Tori Experiment (NCL)')
parser.add_argument('--seed', type=int,default=0, help='seed for rng')

# Runs the Tori experiment with a NCL model
args = parser.parse_args()
#define hyperparams for u,rho,p
beta = 25
#act = lambda x: jax.nn.softplus(x*beta)/beta
act = jnp.sin#lambda x: jax.nn.softplus(x*beta)**2/(beta*beta*2)

#choose spectral frequencies for periodic encodings 
freqs = []
layers = 8
width = 512

if not args.seed:
    seed = np.random.randint(2**32)
else:
    seed = args.seed
key =  random.PRNGKey(seed)
print("Random initial seed:", seed)
x = random.normal(key,shape=(3,))
mlp = MLP(depth=layers,width=width,act=act,out_dim=4,std=1,bias=True)
params = mlp.init(key,embd(x,freqs))

scale = 7e-1
params = jax.tree_map(lambda x: x*scale, params)
params = params.unfreeze()['params']
params = [params, jnp.ones(2)*1e-1]

func_mlp = lambda x,params: mlp.apply({'params':params}, embd(x,freqs))

ncl = NCL(func_mlp,mass_constant=2)

print("Sample NCL output:", ncl(x,params))

#convenience for plotting, only ncl is passed to train/loss module
u = lambda x,params: ncl(x,params)[1:3]/ncl(x,params)[0]
rho = lambda x,params: ncl(x,params)[0]

pde = PDEDivForm()
loss = Loss(ncl)
loss.addTermDom(pde.mom,'mom')
loss.addTermDom(pde.incp,'incp')
loss.addTermInit(pde.init,'init')

gamma = {
    'mom':2e-3,
    'incp':1e-2,
    'init':3e1
}
loss.setGamma(gamma)

sched = optax.piecewise_constant_schedule(init_value=5e-4,
                                    boundaries_and_scales={300:0.5,
                                                          }
                                   )

runToriExperiment(params=params, 
                  key=key,
                  pde=pde,
                  loss=loss,
                  u=u,
                  rho=rho,
                  sched=sched,
                  apx=str(seed)+"ncl",
                  #load_path='training_dumps/3222299688ncl'
                    )




