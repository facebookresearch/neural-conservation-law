"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax.numpy as jnp
import jax
from sampling import CubeSampler
from training import Trainer
from plotting import plotVelDensBall,plotStats
from utils import *
from jax import random, vmap,grad
import numpy as np
from models import MixtureScore,MLP,DivFree,MLP_Skip,DivFreeImplicit
from tqdm import tqdm 
from losses import HelmholtzLoss
from sampling import CubeSampler

#evaluates how well the model is fitting the target div-free field
def evalModel(params, loss, key, dim,N=1000):
    
    error = 0
    curl_error = 0
    div_error = 0
    pot_error = 0
    for i in range(10):
        key,_ = random.split(key,2)
        smp = CubeSampler(dim=dim,N=N)
        pts = smp.smpDom(key,smp.smpTime(key))
        ncl_ev = vmap(lambda x: loss.ncl(x,params))(pts)
        pot_ev = vmap(loss.pot)(pts)
        tar_ev = vmap(loss.targ)(pts)
        error += jnp.sum(jnp.linalg.norm(ncl_ev - tar_ev,axis=1))/(N*10)
        pot_error += jnp.sum(jnp.linalg.norm(ncl_ev - pot_ev,axis=1))/(N*10)
        curl_error += jnp.sum(loss.curl_loss(params,pts))/(10)
        div_error += jnp.sum(loss.div_loss(params,pts))/(10)
    print("Model evaluation at random point:", ncl_ev[0])
    print("Potential evaluation at same point:", pot_ev[0])
    print("Target evaluation at same point:", tar_ev[0])

    return error, pot_error, curl_error, div_error

#helmholtz experiment where we know the decomposition and are fitting to the (known) div-free part
def runHelmholtzExperiment(params, key, model, apx, opt, dim, eps=1e3,load_path='',m_type='curl'):

    if not load_path == '':
        params, opt_st = loadState(load_path)

    smp = CubeSampler(dim=dim,N=1000)
    #plotVelDensBall(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + "_warmup")
    opt_st = opt.init(params)

    key =  random.PRNGKey(42) #fixed key for reproducibility
    #set up potential and div free for target
    # mu = random.normal(key,shape=(2,5))
    # potential = MixtureScore(mu0=mu[0],mu1=mu[1])
    act = jnp.sin
    x = random.normal(key,(dim+1,))
    mlp3 = MLP(depth=2,width=64,act=act,out_dim=1,std=1,bias=True)
    scale = 1.5
    params3 = mlp3.init(key,x[1:])
    params3 = jax.tree_map(lambda x: x*scale, params3)
    params3 = params3.unfreeze()['params']
    potential = lambda x: grad(lambda x: mlp3.apply({'params':params3}, x)[0])(x[1:])
    
    
    #div_free target
    act = jnp.sin
    key, mkey = random.split(key,2)
    x = random.normal(key,(dim+1,))
    mlp2 = MLP(depth=2,width=64,act=act,out_dim=dim*(dim-1)//2,std=1,bias=True)
    params2 = mlp2.init(mkey,x[1:])

    #scales parameters to adjust behavior
    scale = 1.2
    params2 = jax.tree_map(lambda x: x*scale, params2)
    params2 = params2.unfreeze()['params']
    func_mlp2 = lambda x,params: mlp2.apply({'params':params}, x)
    #divfree_targ = lambda x: mlp2.apply({'params':params2}, x[1:])

    # def divfree_targ(x):
    #     ncl = DivFreeImplicit(lambda y,params: func_mlp2(jnp.array([x[0],*y]),params))
    #     return 0.1*ncl(x[1:],params2)
    divfree_targ = lambda x: DivFree(func_mlp2)(x[1:],params2)

    ev_pts = smp.smpDom(key,smp.smpTime(key))
    print(ev_pts.shape)
    print("Norm of random potential evaluation:", jnp.linalg.norm(vmap(potential)(ev_pts[:100]),axis=1)**2)
    print("Norm of 5 random divfree target evaluations:", jnp.linalg.norm(vmap(divfree_targ)(ev_pts[:5]),axis=1)**2)
    print(divfree_targ(ev_pts[0]))
    print("divergence at 5 random points", vmap(lambda x: div(lambda y: divfree_targ(jnp.array([x[0],*y])))(x[1:]))(ev_pts[:5]))
    loss = HelmholtzLoss(potential,divfree_targ,model)
    
    if m_type == 'div':
        loss.loss_term = 'div'
    elif m_type == 'curl':
        loss.loss_term = 'curl'
    print(loss.loss_term)

    trainer = Trainer(opt,loss,smp)

    eval_loss = evalModel(params,loss,key,dim=dim)
    print("Loss to ground truth (init)", eval_loss)

    #tkey,key = random.split(key)
    #full run
    stats = []
    ev_stats = []
    
    for i in range(1000):
        tkey,key = random.split(key)
        params, opt_st,stats = trainer.trainModel(params, tkey, opt_st, stats=stats,steps=int(eps))
        #plotVelDensBall(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + str(i*eps))
        eval_loss = evalModel(params,loss,key,dim=dim)
        print("Loss to ground truth", eval_loss[0],"curl_error",eval_loss[1], "div_error", eval_loss[2])
        ev_stats.append(eval_loss)
        plotStats(stats[5:],apx="helmholtz_experiment_" + apx)
        plotStats(ev_stats[5:],apx="helmholtz_experiment_eval" + apx)
        saveState(params,opt_st,stats,"training_dumps/" + apx)

#helmholtz experiment where we do not know the decomposition, 
def runHelmholtzExperiment2(params, key, model, apx, opt, dim, eps=1e3,load_path=''):

    if not load_path == '':
        params, opt_st = loadState(load_path)

    smp = CubeSampler(dim=dim,N=1000)
    #plotVelDensBall(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + "_warmup")
    opt_st = opt.init(params)

    key =  random.PRNGKey(42) #fixed key for reproducibility
 
    #target network
    act = jax.nn.tanh
    key, mkey = random.split(key,2)
    x = random.normal(key,(dim+1,))
    mlp2 = MLP(depth=2,width=64,act=act,out_dim=dim,std=1,bias=True)
    params2 = mlp2.init(mkey,x[1:])

    #scales parameters to adjust behavior
    scale = 1.5
    params2 = jax.tree_map(lambda x: x*scale, params2)
    params2 = params2.unfreeze()['params']
    func_mlp2 = lambda x,params: mlp2.apply({'params':params}, x)
    targ = lambda x: mlp2.apply({'params':params2}, x[1:])


    ev_pts = smp.smpDom(key,smp.smpTime(key))
    print(ev_pts.shape)
    print("Norm of 5 random target evaluations:", jnp.linalg.norm(vmap(targ)(ev_pts[:5]),axis=1)**2)
    print(targ(ev_pts[0]))
    
    loss = HelmholtzLoss(lambda x: jnp.zeros_like(x[1:]),targ,model)
    loss.loss_term = 'curl'

    trainer = Trainer(opt,loss,smp)

    eval_loss = evalModel(params,loss,key,dim=dim)
    print("Loss to ground truth (init)", eval_loss)
    stats = []
    ev_stats = []
    
    for i in range(1000):
        tkey,key = random.split(key)
        params, opt_st,stats = trainer.trainModel(params, tkey, opt_st, stats=stats,steps=int(eps))
        #plotVelDensBall(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + str(i*eps))
        eval_loss = evalModel(params,loss,key,dim=dim)
        print("Loss to ground truth", eval_loss[0],"curl_error",eval_loss[1], "div_error", eval_loss[2])
        ev_stats.append(eval_loss)
        plotStats(stats[5:],apx="helmholtz_experiment_" + apx)
        plotStats(ev_stats[5:],apx="helmholtz_experiment_eval" + apx)
        saveState(params,opt_st,stats,"training_dumps/" + apx)

#ON TORI
#helmholtz experiment where we know the decomposition and are fitting to the (known) div-free part
def runHelmholtzExperiment3(params, key, model, apx, opt, dim, eps=1e3,load_path='',m_type='curl',N=1000):

    if not load_path == '':
        params, opt_st = loadState(load_path)

    smp = CubeSampler(dim=dim,N=N)
    #plotVelDensBall(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + "_warmup")
    opt_st = opt.init(params)

    key =  random.PRNGKey(42) #fixed key for reproducibility
    #set up potential and div free for target
    # mu = random.normal(key,shape=(2,5))
    # potential = MixtureScore(mu0=mu[0],mu1=mu[1])
    act = jnp.sin
    x = random.normal(key,(dim+1,))
    mlp3 = MLP(depth=2,width=64,act=act,out_dim=1,std=1,bias=True)
    scale = 1.5
    params3 = mlp3.init(key,periodic(x[1:]))
    params3 = jax.tree_map(lambda x: x*scale, params3)
    params3 = params3.unfreeze()['params']
    potential = lambda x: grad(lambda x: mlp3.apply({'params':params3}, periodic(x))[0])(x[1:])
    
    
    #div_free target
    act = jnp.sin
    key, mkey = random.split(key,2)
    x = random.normal(key,(dim+1,))
    mlp2 = MLP(depth=1,width=128,act=act,out_dim=dim*(dim-1)//2,std=1,bias=True)
    params2 = mlp2.init(mkey,periodic(x[1:]))

    #scales parameters to adjust behavior
    scale = 1.1
    params2 = jax.tree_map(lambda x: x*scale, params2)
    params2 = params2.unfreeze()['params']
    func_mlp2 = lambda x,params: mlp2.apply({'params':params}, periodic(x))
    #divfree_targ = lambda x: mlp2.apply({'params':params2}, x[1:])

    # def divfree_targ(x):
    #     ncl = DivFreeImplicit(lambda y,params: func_mlp2(jnp.array([x[0],*y]),params))
    #     return 0.1*ncl(x[1:],params2)
    divfree_targ = lambda x: DivFree(func_mlp2)(x[1:],params2)

    ev_pts = smp.smpDom(key,smp.smpTime(key))
    print(ev_pts.shape)
    print("Norm of random potential evaluation:", jnp.linalg.norm(vmap(potential)(ev_pts[:100]),axis=1)**2)
    print("Norm of 5 random divfree target evaluations:", jnp.linalg.norm(vmap(divfree_targ)(ev_pts[:5]),axis=1)**2)
    print(divfree_targ(ev_pts[0]))
    print("divergence at 5 random points", vmap(lambda x: div(lambda y: divfree_targ(jnp.array([x[0],*y])))(x[1:]))(ev_pts[:5]))
    loss = HelmholtzLoss(potential,divfree_targ,model)
    
    if m_type == 'div':
        loss.loss_term = 'div'
    elif m_type == 'curl':
        loss.loss_term = 'curl'
    print(loss.loss_term)

    trainer = Trainer(opt,loss,smp)

    eval_loss = evalModel(params,loss,key,dim=dim,N=N)
    print("Loss to ground truth (init)", eval_loss)

    #tkey,key = random.split(key)
    #full run
    stats = []
    ev_stats = []
    
    for i in range(1000):
        tkey,key = random.split(key)
        params, opt_st,stats = trainer.trainModel(params, tkey, opt_st, stats=stats,steps=int(eps))
        #plotVelDensBall(lambda x: u(x,params), lambda x: rho(x,params),apx=apx + str(i*eps))
        eval_loss = evalModel(params,loss,key,dim=dim,N=N)
        print("Loss to ground truth", eval_loss[0],"ground truth error (potential)", eval_loss[1], "curl_error",eval_loss[2], "div_error", eval_loss[3])
        ev_stats.append(eval_loss)
        plotStats(stats[5:],apx="helmholtz_experiment_" + apx)
        plotStats(ev_stats[5:],apx="helmholtz_experiment_eval" + apx)
        saveState(params,opt_st,[stats,ev_stats],"training_dumps/" + apx)
