#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
#pixels /inch 
dpi =200

def fig_lin(f,x,sigma,name,xlabel='',ylabel=''):
    plt.figure()
    y = f(x)+np.random.normal(0,sigma,x.shape)
    plt.scatter(x,y,marker='+',color='red',s=50)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('./fig',name))

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

def genere_sample(N,sigma):
    x0 = np.random.normal(-1,sigma,(N,2))
    y0 = np.zeros(x0.shape[0])
    x1 = np.random.normal(1,sigma,(N,2))
    y1 = np.ones(x1.shape[0])
    
    x = np.concatenate((x0,x1),axis=0)
    y = np.concatenate((y0,y1))
    mix = np.arange(2*N)
    np.random.shuffle(mix)
    return(x[mix],y[mix])
    

def h(theta,x):
    theta.shape = (theta.shape[0],1)
    x = x.reshape(theta.shape[0],1)
    scal = np.dot(theta.T,x)
    return 1.0/(1+ np.exp(-scal))

def lmsq(x,y,theta):
    n = y.size
    D = np.array([0.5*(h(theta,x[i,:])-y[i])**2 for i in range(n)])
    return (1.0/n)*np.sum(D)

def log_cost(x,y,theta):
    n = y.size
    D = np.array([(y[i]-1)*np.log(1-h(theta,x[i,:]))-y[i]*np.log(h(theta,x[i,:])) for i in range(n)])
    return (1.0/n)*np.sum(D)

def fig_lmsq():
    plt.figure()
    (x,y) = genere_sample(100,1)
    xv,yv = np.meshgrid(np.linspace(0,7,20),np.linspace(0,7,20))
    J = np.zeros(xv.shape)
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            theta = np.array([xv[i,j],yv[i,j]])
#            J[i,j] = log_cost(x,y,theta)
            J[i,j] = lmsq(x,y,theta)
    plt.imshow(np.log(J))
    plt.gca().invert_yaxis()
    plt.xticks([0])
    plt.yticks([0])
    plt.rc('text', usetex=True)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.savefig('./fig/cost_lmsq.png')

def fig_log():
    plt.figure()
    (x,y) = genere_sample(100,1)
    xv,yv = np.meshgrid(np.linspace(0,6,20),np.linspace(0,6,20))
    J = np.zeros(xv.shape)
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            theta = np.array([xv[i,j],yv[i,j]])
            J[i,j] = log_cost(x,y,theta)
#            J[i,j] = lmsq(x,y,theta)
    plt.imshow(np.log(J+1e-6))
    plt.gca().invert_yaxis()
    plt.xticks([0])
    plt.yticks([0])
    plt.rc('text', usetex=True)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.savefig('./fig/cost_log.png')



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

       
def fig_cost1():
    plt.figure()
    x = np.linspace(0,1,50)
    y = -np.log(x)
    plt.plot(x,y,LineWidth=3)
    plt.rc('text', usetex=True)
    plt.xlabel(r'$h_\theta(x)$')
    plt.ylabel('D')
    plt.xticks([0,1])
    plt.yticks([])
    plt.savefig('./fig/cost1.png')
     
def fig_cost0():
    plt.figure()    
    x = np.linspace(0,1,50)
    y = -np.log(1-x)
    plt.plot(x,y,LineWidth=3)
    plt.rc('text', usetex=True)
    plt.xlabel(r'$h_\theta(x)$')
    plt.ylabel('D')
    plt.xticks([0,1])
    plt.yticks([])
    plt.savefig('./fig/cost0.png')

def trim():
    for filename in os.listdir('./fig/'):
        if filename.endswith('.png'):
            print filename
            cmdline = 'convert fig/'+filename+' -trim fig/'+filename
            os.system(cmdline)

#List of figs
#fig_lin(lambda x:2000*x+1000,9+70*np.random.rand(20),10000,'surf_price',xlabel='Surface',ylabel='Price')
#fig_lin(lambda x:2*x +1, 10*np.random.rand(30),5,'positive')
#fig_lin(lambda x:-3*x +2, 10*np.random.rand(20),1,'negative')
#fig_lin(lambda x:np.exp(x),4*np.random.rand(20),1,'nolinear')
#fig_lin(lambda x:0,6*np.random.rand(20),0.1,'nolink')

trim()
