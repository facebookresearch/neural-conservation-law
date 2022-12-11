"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata as int_grid
from mpl_toolkits.axes_grid1 import make_axes_locatable


#plotting function to generate the figures for the Tori problem
def plotVelDensTori(u,rho,T=[0,0.15,0.3],apx="",colorbar=True):
    num = len(T)
    fig,ax = plt.subplots(2,num,figsize=(21,14))
    
    plotVelsTori(u,T,ax[0])
    plotDensTori(rho,T,ax[1])
    
    fig.savefig("plots/2d_tori_{}.png".format(apx))
    plt.close(fig)

    
def plotVelsTori(u,T,ax1,ax2=None,rf_sols=None):
    N = 250
    a=1
    X,Y = np.meshgrid(np.linspace(0,a,N),np.linspace(0,a,N))
    
    for i,t in enumerate(T):
        pts = jnp.vstack([np.ones(X.reshape(-1).shape)*t,X.reshape(-1),Y.reshape(-1)]).T
        vel = vmap(u)(pts)
        U = np.array(vel[:,0].reshape(X.shape))
        V = np.array(vel[:,1].reshape(Y.shape))
        ax1[i].set_xlim(0,a)
        ax1[i].set_ylim(0,a)
        plt_str = ax1[i].streamplot(X,Y,U,V,density=0.45,arrowsize=4,linewidth=4,color='k')
        ax1[i].axis('off')
        
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k', facecolor='none')
        ax1[i].add_patch(rect)
        
        if ax2 is not None:
            rf = rf_sols[i]
            points = np.array(rf.points[:,:2])
            rho_ref = np.array(rf['rho_n'])
            u_ref = np.array(rf['u_n'])

            rho_ref = int_grid(points,rho_ref,(X,Y))
            u_ref = int_grid(points,u_ref,(X,Y))

            ax2[i].set_xlim(0,a)
            ax2[i].set_ylim(0,a)
            #ax[0].plot(bd[0],bd[1])
            U_1 = np.array(u_ref[:,:,0])
            V_1 = np.array(u_ref[:,:,1])
            plt_str = ax2[i].streamplot(X,Y,U_1,V_1,density=0.45,arrowsize=4,linewidth=4,color='k')
            ax2[i].axis('off')

            rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k', facecolor='none')
            ax2[i].add_patch(rect)

def plotDensTori(rho,T,ax1,ax2=None,rf_sols=None,colorbar=True,clim=None):
    N = 250
    a=1
    X,Y = np.meshgrid(np.linspace(0,a,N),np.linspace(0,a,N))

    for i,t in enumerate(T):
        pts = jnp.vstack([np.ones(X.reshape(-1).shape)*t,X.reshape(-1),Y.reshape(-1)]).T
        dens = vmap(rho)(pts).reshape(X.shape)
        dens = np.array(dens)
        
        ax1[i].set_xlim(0,a)
        ax1[i].set_ylim(0,a)
        if clim:
            plt_dens1 = ax1[i].contourf(X,Y,dens,150,vmin=clim[0],vmax=clim[1])
        else:
            plt_dens1 = ax1[i].contourf(X,Y,dens,150)
        ax1[i].axis('off')
        
        if ax2 is not None:
            rf = rf_sols[i]
            points = np.array(rf.points[:,:2])
            rho_ref = np.array(rf['rho_n'])
            rho_ref = int_grid(points,rho_ref,(X,Y))

            ax2[i].set_xlim(0,a)
            ax2[i].set_ylim(0,a)
            #ax[0].plot(bd[0],bd[1])
            plt_dens2 = ax2[i].contourf(X,Y,rho_ref,150,vmax=clim[1],vmin=clim[0])
            ax2[i].axis('off')
    
    if colorbar:
        divider1 = make_axes_locatable(ax1[-1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.10)
        plt.colorbar(plt_dens1,cax=cax1)

def plotStats(stats,apx):
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    
    ax.plot(stats)
    ax.set_yscale('log')
    ax.set_ylabel('loss')
    ax.set_xlabel('steps (x100)')
    fig.savefig("plots/{}.png".format(apx))
    plt.close(fig)
    
#plotting function to generate the figures for the ball problem
def plotVelDensBall(u,rho,T=[0,0.25,0.5],apx=""):
    box= 8
    #our plots
    fig1,ax1 = plt.subplots(1,3,figsize=(3*box,box))
    fig2,ax2 = plt.subplots(1,3,figsize=(3*box,box))
    
    for i,t in enumerate(T): 
        plotDensBall(t,rho,Z=0,ax=ax1[i])
        plotVelBall(t,u,Z=0,ax=ax2[i])
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig("plots/3d_slice_densplot_ours_{}.png".format(apx))
    fig2.savefig("plots/3d_slice_streamplot_ours_{}.png".format(apx))
    

def plotVelBall(T,u,ax,Z=0):
    N = 250
    a = 1.1
    X,Y = np.meshgrid(np.linspace(-a,a,N),np.linspace(-a,a,N))
    exterior = X**2 + Y**2 + Z**2 >= 1
    pts = jnp.vstack([np.ones(X.reshape(-1).shape)*T,X.reshape(-1),Y.reshape(-1),np.ones(X.reshape(-1).shape)*Z]).T

    #plots the streamplot for the velocity field
    if ax is None:
        fig,ax = plt.subplots(1,2,figsize=(14,7))
    ax.set_xlim(-a,a)
    ax.set_ylim(-a,a)
    
    vel = vmap(u)(pts)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    #mask the outside of the ball
    U[exterior] = np.nan
    V[exterior] = np.nan
    plt_str = ax.streamplot(X,Y,U,V,density=0.35,color=U**2 + V**2, arrowsize=5,linewidth=3)
    
    #add outline for aesthetics
    circle = plt.Circle((0, 0), 1.05, fill=False, lw=3,color='k')
    ax.add_patch(circle)
    ax.axis('off')
    

def plotDensBall(T,rho,ax,Z=0):
    N = 250
    a = 1.1
    X,Y = np.meshgrid(np.linspace(-a,a,N),np.linspace(-a,a,N))
    exterior = X**2 + Y**2 + Z**2 >= 1
    pts = jnp.vstack([np.ones(X.reshape(-1).shape)*T,X.reshape(-1),Y.reshape(-1),np.ones(X.reshape(-1).shape)*Z]).T

    density = vmap(rho)(pts).reshape(X.shape)
    density = np.array(density)
    density[exterior] = np.nan
    plt_dens = ax.contourf(X,Y,density,20)
    circle = plt.Circle((0, 0), 1.0, fill=False, lw=3,color='k')
    ax.add_patch(circle)
    
    ax.set_xlim(-a,a)
    ax.set_ylim(-a,a)
    
    ax.axis('off')

        
        
    
        
        