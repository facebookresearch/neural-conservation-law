"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax.numpy as jnp 
import jax.random as random 

# creates a sampler class to sample points from 
# 1. The interior of the domain 
# 2. The boundary 
# 3. The initial condition interval 
# 3 is here because we want to fuzz the initial condition -- sample at times slightly non-zero 
# This really seems to improve stability in practice 


#base class for samplers (to be extended by subclasses for specific domains)
class Sampler(object):

    #delta is the fuzzing timestep for the initial condition
    def __init__(self,delta=1e-2):
        self.delta = delta
        self.T = 1
        self.bsize = 100

    #N is always the batch size
    #samples from the interior of the domain 
    def smpDom(self,key,t,N=None):
        raise NotImplementedError

    #samples from the boundary
    def smpBd(self,key,N=None):
        raise NotImplementedError

    #samples time
    #defaults to uniform but can be overriden
    def smpTime(self,key,N=None):
        if N is None:
            N = self.N
        return random.uniform(key,shape=(N,1))*self.T

    #fuzzes the initial condition by sampling at different time 
    def smpInit(self,key,N=None):
        if N is None:
            N = self.N
        keys = random.split(key,3)
        t = random.uniform(keys[0],shape=(N,1))*self.delta
        return self.smpDom(keys[1],t)


#sampler class for the periodic square (flat 2-Tori)
class ToriSampler(Sampler):

    #T is the end time for the domain
    #N is the default batch size
    def __init__(self,T=1,N=100):
        super().__init__()
        self.T = T
        self.N = N
    
    #samples from the interior of the periodic square
    def smpDom(self,key,t,N=None):
        if N is None:
            N = self.N
        keys = random.split(key,2)
        
        pts = random.uniform(keys[0],shape=(N,2))

        return jnp.concatenate([t,pts],axis=1)

    #no boundary conditions since we are on a periodic domain 

    def smpBd(self,key,t,N=None):
        if N is None:
            N = self.N
        return jnp.zeros(shape=(N,2))
    
    
#Sampler for the 3d unit ball problem 
class BallSampler(Sampler):
    
    #T is the end time for the domain
    #N is the default batch size
    def __init__(self,T=1,N=100):
        super().__init__()
        self.T = T
        self.N = N
    
    #samples from the interior of the periodic square
    def smpDom(self,key,t,N=None):
        if N is None:
            N = self.N
        keys = random.split(key,2)
        
        pts = self.smpBd(keys[0],t)
        scale = random.uniform(keys[1],shape=(N,1))
        scale = jnp.sqrt(scale)
        pts = pts*scale

        return jnp.concatenate([t,pts[:,1:]],axis=1)

    #no boundary conditions since we are on a periodic domain 

    def smpBd(self,key,t,N=None):
        if N is None:
            N = self.N
            
        pts = random.uniform(key,shape=(N,3))*2 - 1
        pts = pts / jnp.linalg.norm(pts,axis=1).reshape(-1,1)
        return jnp.concatenate([t,pts],axis=1)
    
    

class CubeSampler(Sampler):

    #T is the end time for the domain
    #N is the default batch size
    def __init__(self,dim=3,width=1,N=500):
        super().__init__()
        self.dim = dim
        self.width = 2 #width of cube 
        self.N = N
    
    #samples from the interior of the periodic square
    def smpDom(self,key,t,N=None):
        if N is None:
            N = self.N
        pts = (random.uniform(key,shape=(N,self.dim)) - 0.5)*self.width

        return jnp.concatenate([t,pts],axis=1)
    
    def smpBd(self,key,N=None):
        return 0.