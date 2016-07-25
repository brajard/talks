#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
#pixels /inch 
dpi =200

def fig_sigm():
    """ Figue of sigmoid
    """
    plt.figure()
    x = np.linspace(-5,5,100)
    y = 1.0 / (1+np.exp(-x)) 
    plt.plot(x,y,linewidth=3)
    plt.plot([np.min(x),0],[0.5,0.5],'--k')
    plt.plot([0,0],[-0.01,0.5],'--k')
    plt.ylim([-0.01,1.1])
    plt.xlim([np.min(x),np.max(x)])

    plt.xticks([0])
    plt.yticks([0,0.5,1])
    plt.savefig('./fig/fig_sigm.png',dpi=dpi)


def fig_2cl_nonoise(mu,sigma,name):
    """Figure of linearly separated gaussian"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #mu = [np.array([-1,-1]),np.array([1,1])]
    #sigma = np.array([0.4,0.4])
    color = ['blue','red','blue','red']
    shape = ['o','x','o','x']
    N = 50
    for i in range(len(mu)):
        x = np.random.normal(mu[i][0],sigma,N)
        y = np.random.normal(mu[i][1],sigma,N)
        ax.plot(x,y,color=color[i],marker=shape[i],markersize=8,linestyle='')
    
    plt.savefig(os.path.join('./fig','fig_'+name),dpi=dpi)
      
def trim():
    for filename in os.listdir('./fig/'):
        if filename.endswith('.png'):
            print filename
            cmdline = 'convert fig/'+filename+' -trim fig/'+filename
            os.system(cmdline)

#List of figs
#fig_sigm()
fig_2cl_nonoise([[-1,-1],[1,1]],0.4,'2cl_nonoise')
fig_2cl_nonoise([[-1,-1],[1,1]],1,'2cl_noise')
fig_2cl_nonoise([[-1,-1],[-1,1],[1,1],[1,-1]],0.3,'4cl_nonoise')

trim()
