#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os

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



#List of figs
fig_unif()
fig_gauss()

for filename in os.listdir('./fig/'):
    if filename.endswith('.png'):
        print filename
        cmdline = 'convert fig/'+filename+' -trim fig/'+filename
        os.system(cmdline)
