"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax.numpy as jnp
from jax.numpy import sin,cos,exp
import jax
from sampling import ToriSampler
from training import Trainer
from plotting import plotVelDensTori,plotStats
from utils import *
from jax import random, vmap, grad
import optax
import numpy as np
import os
import pyvista as pv

#define periodic embedding for fixed k
def periodic(x,k):
    c = jnp.pi*2**(k+1)
    return jnp.array([cos(c*x[0]),sin(c*x[0]), cos(c*x[1]),sin(c*x[1])])

#defines full embedding 
def embd(x,freqs=[]):
    return jnp.concatenate([x[:1]] + [periodic(x[1:],0)] + [periodic(x[1:],k) for k in freqs])

#initial condition 
#is called with full time variable 
def rho0_u0(x):
    z = periodic(x[1:],0)
    return jnp.array([1 + (z[1] + z[3])**2, exp(z[3]), 0.5*exp(z[1])])


#clunky but faster than reloading each time
rf_sols = []
for f in os.listdir("./checkpoints/"):
    if str(f)[0] == ".":
        continue
    print(f)
    time = float(str(f).split("checkpoint_T_")[1].split('.vtk')[0])
    f = os.path.join("checkpoints/", f)
    rf_sol = pv.read(f)
    rf_sol['rho_n'] = rf_sol['rho_n']*6 - 5
    rf_sols.append([rf_sol,time])
rf_sols = sorted(rf_sols, key=lambda x: x[1])

#evaluates solution distance to ground truth
def error(key, u, rho):
    t_err = 0
    for (rf,t) in rf_sols:
        pts = jnp.stack([jnp.ones(len(rf.points))*t, rf.points[:,0], rf.points[:,1]]).T
        u_x = vmap(u)(pts)
        rho_x = vmap(rho)(pts)
        
        u_ref = jnp.array(rf['u_n'])[:,:2]
        rho_ref = jnp.array(rf['rho_n'])
        
        u_err = jnp.sum((u_x - u_ref)**2) / (len(pts)*2)
        rho_err = jnp.sum((rho_x - rho_ref)**2) / len(pts)
        
        #if deb: print("Error at time {:3.2f} was rho: {:7.4f} u_n: {:7.4f}".format(t,u_err,rho_err))
        t_err += rho_err #you changed this
    return t_err / len(rf_sols)
        

def runToriExperiment(params, key, u, rho, apx, loss, pde, sched,load_path=''):
    #define pde
    pde.setInitial(rho0_u0)
    
    opt = optax.adam(learning_rate=sched)
    opt_st = opt.init(params)
    
    if not load_path == '':
        params = loadState(load_path)

    smp = ToriSampler(T=0.35,N=1000)
    trainer = Trainer(opt,loss,smp)
    trainer.log_rate = 100
    plotVelDensTori(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + "_warmup")

    #tkey,key = random.split(key)
    #full run
    eps=1e3
    stats = []
    er_stats = []
    for i in range(600):
        tkey,key = random.split(key)
        params, opt_st,stats = trainer.trainModel(params, tkey, opt_st, stats=stats,steps=int(eps))
        plotVelDensTori(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + str(i*eps))
        plotStats(stats[5:],apx="2d_tori_stats_" + apx)
        eval_er = error(key,lambda x: u(x,params), lambda x: rho(x,params))
        print("Error to ground truth:",eval_er)
        er_stats.append(eval_er)
        saveState(params,opt_st,[stats,er_stats],"training_dumps/" + apx)




