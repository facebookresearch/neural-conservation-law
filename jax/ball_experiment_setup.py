"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax.numpy as jnp
from jax.numpy import sin,cos,exp
import jax
from sampling import BallSampler
from training import Trainer
from plotting import plotVelDensBall,plotStats
from utils import *
import jax.random as random
import optax
import numpy as np

#initial condition 
#is called with full time variable 
def rho0_u0(x):
    return jnp.array([3/2 - (x[1]**2 + x[2]**2+ x[3]**2),-2,x[1]-1,0.5])

def runBallExperiment(params, key, u, rho, apx, loss, pde, sched,load_path=''):
    #define pde
    pde.setInitial(rho0_u0)
    
    opt = optax.adam(learning_rate=sched)
    opt_st = opt.init(params)
    
    if not load_path == '':
        params, opt_st = loadState(load_path)

    smp = BallSampler(T=0.5,N=1000)
    trainer = Trainer(opt,loss,smp)
    plotVelDensBall(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + "_warmup")

    #tkey,key = random.split(key)
    #full run
    eps=1e3
    stats = []
    for i in range(10):
        tkey,key = random.split(key)
        params, opt_st,stats = trainer.trainModel(params, tkey, opt_st, stats=stats,steps=int(eps))
        plotVelDensBall(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + str(i*eps))
        plotStats(stats[5:],apx="3d_ball_experiment_" + apx)
        saveState(params,opt_st,stats,"training_dumps/" + apx)




