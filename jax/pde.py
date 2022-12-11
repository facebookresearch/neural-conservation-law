"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax.numpy as jnp 
from jax import jacfwd, grad, jit, vmap
from utils import div

# this is the standard PDE loss for variable density Euler, formulated by assuming the
# velocity - u: R^{n+1} -> R^n 
# density - ρ: R^{n+1} -> R
# pressure - p:R^{n+1} -> R
# following convetion u = (t,x) [time always first]
# are all given separately. So good to use with regular PINN or div-free, but not with NCL
class PDE(object):
    # init takes boundary normal callable as argument
    def __init__(self):
        self.normal = None
        self.initial = None
    
    #set the initial condition for the pde
    def setInitial(self,init):
        self.initial = init
    
    def setNormal(self,nm):
        self.normal = nm
        
    def unpackPINN(self,pinn):
        #curry the u,rho,p functions at the parameters
        u = lambda x: pinn(x)[1:-1]
        rho = lambda x: pinn(x)[0]
        p = lambda x: pinn(x)[-1]
        return u,rho,p


    # Takes u,rho,p callables and evaluates the momentum equation 'rho*u_t + rho*[Du]u + grad(p)' at x
    # Du is spatial jacobian (excluding time)
    def mom(self,pinn,x):
        u,rho,p = self.unpackPINN(pinn)
        Du = jacfwd(u)(x)
        u_t = Du[:,0]
        Du_u = Du[:,1:]
        nabla_p = grad(p)(x)[1:] #chop off time derivative
        rho_x = rho(x)

        return rho_x*u_t + rho_x*Du_u + nabla_p
    
    # Evaluates the continuity eq
    def cont(self,pinn,x):
        u,rho,p = self.unpackPINN(pinn)
        nabla_rho = grad(rho)(x) #full gradient (inc time)
        rho_t = nabla_rho[0]
        div_rhou = div(lambda x: rho(x)*u(x))(x)
        return rho_t + div_rhou
    
    # Evaluates the incompressibility condition
    def incp(self,pinn,x):
        u,rho,p = self.unpackPINN(pinn)
        return div(u)(x)
    
    # evaluates the free-slip boundary condition
    def bd(self,pinn,x):
        u,rho,p = self.unpackPINN(pinn)
        return jnp.dot(u(x),self.normal(x))

    # evaluates the initial condition [rho - rho_0, u - u_0]
    def init(self,pinn,x):
        u,rho,p = self.unpackPINN(pinn)
        return jnp.array([rho(x),*u(x)]) - self.initial(x)



# this is the modified PDE loss for variable density Euler, formulated by assuming the
# vedens - v: R^{n+1} -> R^{n+1}  
# pressure - p:R^{n+1} -> R
# following convetion v = v(t,x) [time always first]
# it evaluates pdes scalable by the density [see appendix A of paper]
class PDEDivForm(object):
    # init takes boundary normal callable as argument
    def __init__(self):
        self.normal = None
        self.initial = None
    
    def setInitial(self,init):
        self.initial = init
    
    def setNormal(self,nm):
        self.normal = nm

    # Takes v,p callables 'rho*u_t + rho*[Du]u + grad(p)' at x
    # Du is spatial jacobian (excluding time)
    def mom(self,v,x):
        v_x = v(x)[:-1]
        rho = v_x[0]
        rho_u = v_x[1:]

        Dv = jacfwd(v)(x)

        nabla_rho = Dv[0]
        Drhou = Dv[1:-1]

        rho3u_t = Drhou[:,0]*rho**2 - nabla_rho[0]*rho*rho_u

        rho3Duu = rho*Drhou[:,1:]@rho_u - jnp.outer(nabla_rho[1:],rho_u).T@rho_u

        nabla_p = Dv[-1,1:] #spatial gradient
        return rho3u_t + rho3Duu + rho**2*nabla_p
    
    
    # Evaluates the (scaled) incompressibility condition
    def incp(self,v,x):
        nabla_rho = grad(lambda y: v(y)[0])(x)
        return jnp.dot(nabla_rho,v(x)[:-1])
    
    # evaluates the free-slip boundary condition
    def bdry(self,v,x):
        return jnp.dot(v(x)[1:-1],self.normal(x))
    
    # evaluates the initial condition [rho - rho_0, rho*u - rho_0*u_0]
    def init(self,v,x):
        in_ev = self.initial(x)
        in_ev = jnp.array([in_ev[0],*(in_ev[1:]*in_ev[0])])
        return v(x)[:-1] - in_ev

    