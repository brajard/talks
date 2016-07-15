#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os

def fig_unif():
    """ Figue uniform density probability function
    """
    x = [-2,2]
    a = -1
    b = 1
    y = 1./(b-a)
    plt.plot([x[0],a,a,b,b,x[1]],[0,0,y,y,0,0],color='red',linewidth=3)
    plt.ylim([-0.01,1.2/(b-a)])
    plt.savefig('./fig/fig_unif.png',dpi=400)


#List of figs
fig_unif()
for filename in os.listdir('./fig/'):
    if filename.endswith('.png'):
        print filename
        cmdline = 'convert fig/'+filename+' -trim fig/'+filename
        os.system(cmdline)
