"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax.numpy as jnp 
from jax import jacfwd, grad, jit, vmap
from utils import div,curl

#
# class LossStd:
#     #initialization incorporates u,rho,p callables as stored methods
#     def __init__(self,pinn,pde,norm=None):
        
#         self.pinn = pinn

#         self.pde = pde
#         self.gamma = jnp.ones(5) #weighting terms for pde_loss

#         # norm applied to residuals 
#         # defaults to squared l2, but l1 also possible
#         if norm is None:
#             self.norm = lambda vec: jnp.sum(jnp.power(vec,2))
#         else:
#             self.norm = norm
    
#     def setGamma(self,gamma):
#         self.gamma = gamma
        
#     # standard batched loss
#     # params: [u_parmas, rho_params, p_params]
#     # dom_pts: interior points to evaluate pde residuals at
#     # bd_pts: boundary points to evaluate pde residuals at
#     # init_pts: initial points to evaluate pde residuals at
#     def lossBatch(self, params,dom_pts,bd_pts,init_pts,debug=False):

#         pde = self.pde

#         #curry the u,rho,p functions at the parameters
#         u = lambda x: self.pinn(x,params)[1:3]
#         rho = lambda x: self.pinn(x,params)[0]
#         p = lambda x: self.pinn(x,params)[3]

#         #evaluate equations
#         mom_ev = vmap(lambda x: pde.mom(u,rho,p,x))(dom_pts)
#         cont_ev = vmap(lambda x: pde.cont(u,rho,x))(dom_pts)
#         incp_ev = vmap(lambda x: pde.incp(u,x))(dom_pts)

#         #evaluate initial condition 
#         init_ev = vmap(lambda x: pde.init(u,rho,x))(init_pts)

#         #evaluate boundary condition
#         bd_ev = vmap(lambda y: pde.bd(u,y))(bd_pts)

#         #apply norm to loss terms 
#         mom_l = self.norm(mom_ev)
#         cont_l = self.norm(cont_ev)
#         incp_l = self.norm(incp_ev)

#         init_l = self.norm(init_ev)
#         bd_l = self.norm(bd_ev)

#         if debug:
#             print("Loss components: (Momentum) (Cont) (Div) (Init) (Boundary)")
#             print(mom_l,cont_l, incp_l, init_l, bd_l)

#         loss_vec = jnp.array([mom_l,cont_l,incp_l,init_l,bd_l])

#         loss_total = jnp.dot(loss_vec, self.gamma) # apply weighted sum and contract

#         return loss_total 
    
class Loss:
    #initialization incorporates u,rho,p callables as stored methods
    def __init__(self,pinn,norm=None):
        
        self.pinn = pinn

        self.gamma = {} #weighting terms for pde_loss
        
        self.termsDom = {}
        self.termsBd = {}
        self.termsInit = {}

        # norm applied to residuals 
        # defaults to squared l2, but l1 also possible
        if norm is None:
            self.norm = lambda vec: jnp.sum(jnp.power(vec,2))
        else:
            self.norm = norm
    
    def setGamma(self,gamma):
        self.gamma = gamma
    
    #adds a term *depending on interior points* to the loss equation
    #term is a callable that takes the pinn callable as an argument
    #name is the string name for associating gamma / debugging
    def addTermDom(self,term,name):
        self.termsDom[name] = term
    
    def addTermBd(self,term,name):
        self.termsBd[name] = term
        
    def addTermInit(self,term,name):
        self.termsInit[name] = term
        
    # standard batched loss
    # params: [u_parmas, rho_params, p_params]
    # dom_pts: interior points to evaluate pde residuals at
    # bd_pts: boundary points to evaluate pde residuals at
    # init_pts: initial points to evaluate pde residuals at
    def lossBatch(self, params,dom_pts,bd_pts,init_pts,debug=False):
        
        loss_ev = {}
        pinn = lambda x: self.pinn(x,params)
        #evaluate domain equations
        for name in self.termsDom:
            term = self.termsDom[name]
            loss_ev[name] = vmap(lambda x: term(pinn,x))(dom_pts)
        
        #evaluate boundary equations
        for name in self.termsBd:
            term = self.termsBd[name]
            loss_ev[name] = vmap(lambda x: term(pinn,x))(bd_pts)
        
        #evaluate initial equations
        for name in self.termsInit:
            term = self.termsInit[name]
            loss_ev[name] = vmap(lambda x: term(pinn,x))(init_pts)
        
        loss_vals = {}
        for name in loss_ev:
            loss_vals[name] = self.norm(loss_ev[name]) / len(loss_ev[name])

        if debug:
            print("Loss components:")
            print("".join(["{}: {:.2e}    ".format(nm,loss_vals[nm]) for nm in loss_vals]))

        loss_total = 0
        
        for name in loss_vals.keys():
            loss_total += loss_vals[name]*self.gamma[name]

        return loss_total 
    


#sets up the loss function for the helmholtz problem

class HelmholtzLoss:
    #potential: potential part of target 
    #div_free_target: divergence free part of 
    def __init__(self, potential, div_free_target, ncl):
        self.pot = potential 
        self.targ = div_free_target
        self.ncl = ncl
        self.loss_term = 'l2'
    
    def div_loss(self,params,pts):
        field = lambda x: self.targ(x) + self.pot(x) - self.ncl(x,params)
        
        div_fn = lambda x: jnp.abs(div(lambda y: field(jnp.array([x[0],*y])))(x[1:]))
        
        return jnp.sum(vmap(div_fn)(pts))/len(pts)
    
    
    def curl_loss(self,params,pts):
        field = lambda x: self.targ(x) + self.pot(x) - self.ncl(x,params)
        
        curl_fn = lambda x: curl(lambda y: field(jnp.array([x[0],*y])))(x[1:])
        
        return jnp.sum(vmap(curl_fn)(pts))/len(pts)
    
    
    def lossBatch(self, params,dom_pts,bd_pts,init_pts,debug=False,**kwargss):
        
        if self.loss_term == 'curl':
            return self.curl_loss(params,dom_pts)
        elif self.loss_term == 'div':
            return self.div_loss(params,dom_pts)
        ncl = lambda x: self.ncl(x,params)

        field = lambda x: self.targ(x) + self.pot(x)
        field_ev = vmap(field)(dom_pts) 
        ncl_ev = vmap(ncl)(dom_pts)
        loss_val = jnp.sum((field_ev - ncl_ev)**2) / len(dom_pts)

        return loss_val


    

        

