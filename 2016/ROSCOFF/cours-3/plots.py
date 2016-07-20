#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
#pixels /inch 
dpi =200

def fig_unif():
    """ Figue uniform density probability function
    """
    plt.figure()
    x = [-2,2]
    a = -1
    b = 1
    y = 1./(b-a)
    plt.plot([x[0],a,a,b,b,x[1]],[0,0,y,y,0,0],color='red',linewidth=3)
    plt.ylim([-0.01,1.2/(b-a)])
    plt.savefig('./fig/fig_unif.png',dpi=dpi)


def fig_gauss():
    """ Figure of the standard normal distribution"""
    plt.figure()
    x = np.linspace(-3,3,100)
    y = np.sqrt(1.0/(2*np.pi))*np.exp(-0.5*x**2)
    plt.plot(x,y,linewidth=2)
    plt.ylim([0,0.42])
    plt.savefig('./fig/fig_gauss.png',dpi=dpi)


def fig_gauss2():
    """ Figure of multivariate normal distribution"""

    fig=plt.figure()
    x, y = np.mgrid[-1.0:1.0:40j, -1.0:1.0:40j]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])

    mu = np.array([0.0, 0.0])

    sigma = np.array([.3, .5])
    covariance = np.diag(sigma**2)

    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    
    # Reshape back to a (30, 30) grid.
    z = z.reshape(x.shape)

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=cm.coolwarm)
    plt.savefig('./fig/fig_gauss2.png',dpi=dpi)
        
def fig_bodyt():
    fig=plt.figure()
    ax = fig.add_subplot(111)
    mu = [37,38.5]
    std = [0.15,0.8]
    color = ['green','red']
    label = ['healthy','sick']
    T = np.linspace(35,40,200)
    for i in range(len(mu)):
        ax.plot(T,multivariate_normal.pdf(T,mean=mu[i],cov=std[i]**2),color=color[i],linewidth=3,label=label[i])
    ax.legend()
    ax.set_xlabel('Body Temperature')
    ax.set_ylabel('Value of the density function')
    plt.savefig('./fig/fig_bodyt.png',dpi=dpi)



def an_sick(y=37.4):
    mu = [37,38.5]
    std = [0.15,0.8]
    p = [0.9,0.1]
    f1 = multivariate_normal.pdf(y,mean=mu[1],cov=std[1]**2)
    f0 = multivariate_normal.pdf(y,mean=mu[0],cov=std[0]**2)
    
    fy = f1*p[1] + f0*p[0]
    return(f1*p[1] / fy)

def fig_sick():
    y = np.linspace(36,40,100)
    P = an_sick(y)
    plt.figure()
    plt.plot(y,P,linewidth=2)
    plt.ylim([-0.01,1.01])
    plt.xlabel('Body''s temperature')
    plt.ylabel('Probability of being sick')
    plt.savefig('./fig/fig_sick.png',dpi=dpi)
    print "probability knowing y=37.4=",an_sick(37.4)
    

def trim():
    for filename in os.listdir('./fig/'):
        if filename.endswith('.png'):
            print filename
            cmdline = 'convert fig/'+filename+' -trim fig/'+filename
            os.system(cmdline)
def fig_entropy():
    p = np.linspace(0.001,0.999,100)
    h = np.zeros(p.shape)
    h= (p-1)*np.log2(1-p) - p*np.log2(p)
    plt.figure()
    plt.plot(p,h,linewidth=2)
    plt.ylim([0,1.05])
    plt.xlabel('p')
    plt.ylabel('H(X)')
    plt.savefig('./fig/fig_entropy.png')

#List of figs
#fig_unif()
#fig_gauss()
#fig_gauss2()
#fig_bodyt()
#fig_sick()
fig_entropy()
trim()
