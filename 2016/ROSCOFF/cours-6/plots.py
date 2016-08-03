#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

plt.close("all")

#pixels /inch 
dpi =200
#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducability
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.figure()
plt.plot(data['x'],data['y'],'+',markersize=10)
plt.savefig('./fig/scatter.png',dpi=dpi)
#plt.show()

for i in range(2,16):  #power of 1 is already there
    colname = 'x_%d'%i      #new var will be x_power
    data[colname] = data['x']**i

from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
    #initialize predictors:
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
    
    #Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered power
    if power in models_to_plot:
        #plt.subplot(models_to_plot[power])
        #plt.tight_layout()
        plt.figure()
        plt.plot(data['x'],data['y'],'.',markersize=15)
        plt.plot(data['x'],y_pred,color='red',linewidth=2)
        #plt.title('power: %d'%power)
        plt.savefig('./fig/linreg_pow' + models_to_plot[power] + '.png')
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret

#Initialize a dataframe to store the results:
col = ['rmse','th_0'] + ['th_%d'%i for i in range(1,16)]
ind = ['max_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define the powers for which a plot is required:
models_to_plot = {1:'1',3:'3',6:'4',9:'9',12:'12',15:'15'}

#Iterate through all powers and assimilate results
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)

with open('coeflin.tex','w') as f:
    f.write(coef_matrix_simple.to_latex())    


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

def trim():
    for filename in os.listdir('./fig/'):
        if filename.endswith('.png'):
            print filename
            cmdline = 'convert fig/'+filename+' -trim fig/'+filename
            os.system(cmdline)

#List of figs
#fig_sigm()
#fig_2cl_nonoise([[-1,-1],[1,1]],0.4,'2cl_nonoise')
#fig_2cl_nonoise([[-1,-1],[1,1]],1,'2cl_noise')
#fig_2cl_nonoise([[-1,-1],[-1,1],[1,1],[1,-1]],0.3,'4cl_nonoise')
#fig_lmsq()
#fig_log()
#fig_cost0()
#fig_cost1()

#trim()

