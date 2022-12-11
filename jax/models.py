"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from flax import linen as nn
import jax.numpy as jnp
import jax
from jax import jacfwd, grad, vmap
import pickle
from utils import div
import jax.scipy as jsp

class MLP(nn.Module):
    depth: int
    width: int
    out_dim: int
    std: float #scaling factor for initialization
    act: callable
    bias: bool

    def setup(self):
        self.layers = [nn.Dense(self.width,use_bias=self.bias) for _ in range(self.depth)] + [nn.Dense(self.out_dim,use_bias=self.bias)]
            
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
        return x
    
class MLP_Skip(nn.Module):
    depth: int
    width: int
    out_dim: int
    std: float #scaling factor for initialization
    act: callable
    bias: bool

    def setup(self):
        self.layers = [nn.Dense(self.width,use_bias=self.bias) for _ in range(self.depth)] + [nn.Dense(self.out_dim,use_bias=self.bias)]
        self.skip = [nn.Dense(self.width,use_bias=self.bias) for _ in range(self.depth)] + [nn.Dense(self.out_dim,use_bias=self.bias)]
            
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(zip(self.layers,self.skip)):
            x = lyr[0](x) + lyr[1](inputs)
            if i != len(self.layers) - 1:
                x = self.act(x)
        return x
    

#class that parameterizes the NCL using the matrix form
class NCL(object):
    def __init__(self,network,mass_constant=2):
        self.network = network
        self.mc = mass_constant
        
    #return type of NCL is [rho,rho u, p] (note middle!)
    def __call__(self,x,params):
        
        def A(x):
            u_v = self.network(x,params[0])[:-1]
            N = len(x)
            A = jnp.zeros((N,N))
            idx = jnp.triu_indices(N,1)
            A = A.at[idx].set(u_v)

            return A - A.T
        u_v = self.network(x,params[0])
        return jnp.array([*div(A)(x),u_v[3]]) + jnp.array([self.mc, *params[1],0]) 

#class that parameterizes the NCL using the matrix form
class NCL_sparse(object):
    def __init__(self,network,mass_constant=2):
        self.network = network
        self.mc = mass_constant
        
    #return type of NCL is [rho,rho u, p] (note middle!)
    def __call__(self,x,params):
        
        def A(x):
            u_v = self.network(x,params[0])[:-1]
            A = jnp.diag((u_v*jnp.roll(x,-1))[:-1],k=1)
            A = A.at[0,-1].set(x[0]*u_v[-1]) 

            return A - A.T
        u_v = self.network(x,params[0])
        return jnp.array([*div(A)(x),u_v[3]]) + jnp.array([self.mc, *params[1],0]) 
        
#class that parameterizes the NCL using the vector form
class NCLImplicit(object):
    def __init__(self,network):
        self.network = network
        
    #return type of NCL is [rho,rho u, p] (note middle!)
    def __call__(self,x,params):
        
        def A(x):
            u = lambda x: self.network(x,params)[:-1]
            A = jacfwd(u)(x)
            return A - A.T
        u_x = self.network(x,params)
        return jnp.array([*div(A)(x),u_x[-1]])
    
#utility bump function
def bump(x):
    dim = x.shape[0]
    const = dim*5
    bump_inner = lambda x: jnp.exp(-1/(1-jnp.dot(x,x)/const))
    return jax.lax.cond(jnp.dot(x,x) < const - 1e-3, bump_inner,lambda x: 0., x)
    
#class that parameterizes arbitrary div_free net
class DivFree(object):
    def __init__(self,network):
        self.network = network
        
    def __call__(self,x,params):
        
        def A(x):
            u_v = self.network(x,params)
            N = len(x)
            
            A = jnp.zeros((N,N))
            idx = jnp.triu_indices(N,1)
            A = A.at[idx].set(u_v) 
            #A = A*jnp.roll(x,1).reshape(-1,1)

            return A - A.T
        return div(A)(x)
#class that parameterizes arbitrary div_free net
class DivFreeSparse(object):
    def __init__(self,network):
        self.network = network
        
    def __call__(self,x,params):
        
        def A(x):
            u_v = self.network(x,params)
            A = jnp.diag((u_v*jnp.roll(x,-1))[:-1],k=1)
            A = A.at[0,-1].set(x[0]*u_v[-1]) 

            return A - A.T
        return div(A)(x)
        
        
class DivFreeImplicit(object):
    def __init__(self,network):
        self.network = network
        
    def __call__(self,x,params):
        
        def A(x):
            u = lambda x: self.network(x,params)
            A = jacfwd(u)(x)
            return A - A.T
        return div(A)(x)
    
#class for the mixture of gaussians score
class MixtureScore:
    def __init__(self,mu0,mu1):
        self.mu0 = mu0
        self.mu1 = mu1
        self.sig0 = 1
        self.sig1 = 1e-2

    def sigma(self,t):
        
        return self.sig0*(1-t)**2 + self.sig1*t**2

    def __call__(self,x):
        #log_pdf without constants
        t = x[0]
        x = x[1:]
        mu_total = self.mu0*(1-t) + self.mu1*t
        

        log_pdf = lambda x: jnp.sum(vmap(lambda mu: jsp.stats.multivariate_normal.logpdf(x,mean=mu,cov=jnp.sqrt(self.sigma(t))))(mu_total))
        return grad(log_pdf)(x)

        
        
               
        
        
        

