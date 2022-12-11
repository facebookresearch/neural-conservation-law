"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax.numpy as np
from jax import grad, vmap, value_and_grad,jit
from jax import random
import optax
from functools import partial
from tqdm import tqdm

def printStats():
    pass

class Trainer():

    def __init__(self, opt, loss_obj, smp):

        self.opt = opt
        self.loss = loss_obj
        self.smp = smp
        self.log_rate  = 100 #steps to avg/print loss

    #set the optimizer 
    #used to change lr, hyper parameters etc
    def set_opt(self,opt):
        self.opt = opt

    @partial(jit,static_argnums=(0,))
    def train_step(self, params, key, opt_st):
        vg_loss = jit(value_and_grad(self.loss.lossBatch))
        smp = self.smp
        keys = random.split(key,5)
        key = keys[0]

        t = smp.smpTime(keys[1])
        dom_pts = smp.smpDom(keys[2],t)
        bd_pts = smp.smpBd(keys[3],t)
        init_pts = smp.smpInit(keys[4])

        lval, lgrad = vg_loss(params,dom_pts,bd_pts,init_pts)
        update, opt_st = self.opt.update(lgrad,opt_st,params)
        params = optax.apply_updates(params, update)
        return params, opt_st, lval
    

    #trains the model given in the loss obj
    
    def trainModel(self, params, key, opt_st,stats=[],steps=int(1e4),hyper_debug=False):

        run_loss = 0
        bar = tqdm(range(steps))
        for i in bar:
            key,_ = random.split(key,2)
            params, opt_st, lval = self.train_step(params, key, opt_st)
            run_loss += lval.item() / self.log_rate
            if hyper_debug:
                print(i,lval.item())
            if not ((i + 1) % self.log_rate):
                #print("-"*10)
                #print("Steps ", i, "avg_loss", run_loss)
                bar.set_description("avg_loss:{:f}".format(run_loss))
                stats.append([run_loss])
                run_loss = 0
                #_ = self.loss.lossBatch(params,dom_pts,bd_pts,init_pts,debug=True) #requires a recomputation :(
        return params, opt_st, stats