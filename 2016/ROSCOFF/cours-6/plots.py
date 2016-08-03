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
xt = ((300-60)*np.random.random(30) + 60)*np.pi/180
yt = np.sin(xt) + np.random.normal(0,0.15,len(xt))
test = pd.DataFrame(np.column_stack([xt,yt]),columns=['x','y'])
plt.figure()
plt.plot(data['x'],data['y'],'+',markersize=10)
plt.savefig('./fig/scatter.png',dpi=dpi)

plt.figure()
plt.plot(data['x'],data['y'],'+',
         markersize=10,color='blue',label='learning')
plt.plot(test['x'],test['y'],'.',
         markersize=10,color='red',label='test')
plt.legend()
plt.savefig('./fig/scatter_test.png',dpi=dpi)
#plt.show()

################ LINEAR ############################

for i in range(2,16):  #power of 1 is already there
    colname = 'x_%d'%i      #new var will be x_power
    data[colname] = data['x']**i
    test[colname] = test['x']**i
    
    
from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot,test=None):
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
        plt.savefig('./fig/linreg_pow' + models_to_plot[power] + '.png',dpi=dpi)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    if test is not None:
        y_test = linreg.predict(test[predictors])
        test_rss = sum((y_test - test['y'])**2)
        ret.extend([test_rss])
    return ret

#Initialize a dataframe to store the results:
col = ['rmse','th_0'] + ['th_%d'%i for i in range(1,16)] + ['rms_test']
ind = ['max_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define the powers for which a plot is required:
models_to_plot = {1:'1',2:'2',3:'3',6:'4',9:'9',12:'12',15:'15'}

#Iterate through all powers and assimilate results
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,range(i+2)+[17]] = linear_regression(data, power=i, models_to_plot=models_to_plot,test=test)

with open('coeflin.tex','w') as f:
    f.write(coef_matrix_simple.iloc[:,0:10].to_latex(float_format=lambda x:"{:+.2e}".format(x)))    

plt.figure()
plt.semilogy(range(1,16),np.abs(coef_matrix_simple['th_1']),linewidth=2)
plt.xlabel('max_pow')
plt.ylabel('|th_1 value|')
plt.savefig('./fig/coefs_th1.png',dpi=dpi)

################## RIDGE ###########################

from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={},test=None):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        #plt.subplot(models_to_plot[power])
        #plt.tight_layout()
        plt.figure()
        plt.plot(data['x'],data['y'],'.',markersize=15)
        plt.plot(data['x'],y_pred,color='red',linewidth=2)
        #plt.title('power: %d'%power)
        plt.savefig('./fig/ridge_alpha' + models_to_plot[alpha] + '.png',dpi=dpi)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    if test is not None:
        y_test = ridgereg.predict(test[predictors])
        test_rss = sum((y_test - test['y'])**2)
        ret.extend([test_rss])
    return ret


#Initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Set the different values of alpha to be tested
alpha_ridge = [0, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
na = len(alpha_ridge)
#Initialize the dataframe for storing coefficients.
#col = ['rss','t_0'] + ['coef_x_%d'%i for i in range(1,16)]
col = ['rmse','th_0'] + ['th_%d'%i for i in range(1,16)] + ['rms_test']
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,na)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {0:'0',1e-15:'1e-15', 1e-10:'1e-10', 1e-4:'1e-4', 1e-3:'1e-3', 1e-2:'1e-2', 5:'5'}
for i in range(na):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot,test)



with open('coefridge.tex','w') as f:
    f.write(coef_matrix_ridge.iloc[:,0:10].to_latex(float_format=lambda x:"{:+.2e}".format(x)))    

plt.figure()
plt.semilogy(range(0,na),np.abs(coef_matrix_ridge['th_5']),linewidth=2)
plt.xticks(range(0,na),[str(alpha) for alpha in alpha_ridge])
plt.xlabel('alpha')
plt.ylabel('|th_5 value|')
plt.savefig('./fig/coefs_th5_ridge.png',dpi=dpi)


#################### LASSO #######################
from sklearn.linear_model import Lasso
def lasso_regression(data, predictors, alpha, models_to_plot={},test=None):
    #Fit the model
    if alpha==0:
        lassoreg = LinearRegression(normalize=True)
    else:
        lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        #plt.subplot(models_to_plot[power])
        #plt.tight_layout()
        plt.figure()
        plt.plot(data['x'],data['y'],'.',markersize=15)
        plt.plot(data['x'],y_pred,color='red',linewidth=2)
        #plt.title('power: %d'%power)
        plt.savefig('./fig/lasso_alpha' + models_to_plot[alpha] + '.png',dpi=dpi)
      
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    if test is not None:
        y_test = lassoreg.predict(test[predictors])
        test_rss = sum((y_test - test['y'])**2)
        ret.extend([test_rss])
    return ret

#Initialize predictors to all 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Define the alpha values to test
alpha_lasso = [0, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
nl = len(alpha_lasso)

#Initialize the dataframe for storing coefficients.
#col = ['rss','t_0'] + ['coef_x_%d'%i for i in range(1,16)]
col = ['rmse','th_0'] + ['th_%d'%i for i in range(1,16)] + ['rms_test']
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,nl)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {0:'0',1e-15:'1e-15', 1e-10:'1e-10', 1e-4:'1e-4', 1e-3:'1e-3', 1e-2:'1e-2', 5:'5'}

#Iterate over the 10 alpha values:
for i in range(nl):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot,test)


with open('coefridge.tex','w') as f:
    f.write(coef_matrix_lasso.iloc[:,0:10].to_latex(float_format=lambda x:"{:+.2e}".format(x)))    

plt.figure()
plt.semilogy(range(0,na),np.abs(coef_matrix_ridge['th_5']),linewidth=1,color='blue',label='ridge')
plt.semilogy(range(0,nl),np.abs(coef_matrix_lasso['th_5']),linewidth=2,color='red',label='lasso')
plt.legend()
plt.xticks(range(0,nl),[str(alpha) for alpha in alpha_lasso])
plt.xlabel('alpha')
plt.ylabel('|th_5 value|')
plt.savefig('./fig/coefs_th5_lasso.png',dpi=dpi)


################### LEARNING / TEST #################
plt.figure()
plt.semilogy(range(0,na),coef_matrix_ridge['rmse']/data.shape[0],'+-',
         color='blue',linewidth='2',markersize=10,label='training')
plt.semilogy(range(0,na),coef_matrix_ridge['rms_test']/test.shape[0],'+-',
         color='red',linewidth='2',markersize=10,label='test')
plt.legend()
plt.xticks(range(0,nl),[str(alpha) for alpha in alpha_ridge])
plt.xlabel('alpha')
plt.ylabel('error')
plt.savefig('./fig/validation.png',dpi=dpi)


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

trim()

