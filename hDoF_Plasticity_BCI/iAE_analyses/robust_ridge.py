# -*- coding: utf-8 -*-
"""
@author: nikic
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random as rnd
import numpy.linalg as lin
import os
import scipy.stats as stats



#### robust ridge function ####
def robust_ridge(x,y,k):
    
    # setting things up    
    x = stats.zscore(x,axis=0)
    y = stats.zscore(y,axis=0)
    w = np.ones((len(y),1))
    W = np.diagflat(w)
    
    # run robust ridge regression once to get residuals using Huber loss for 95% CI of Gaussian Median
    bhat = lin.inv(x.T @ W @ x + k*np.eye(x.shape[1])) @ (x.T @ W @ y)
    yhat = x@bhat
    r = y-yhat
    g = np.zeros((len(w),1))
    m1 = stats.median_abs_deviation(r)
    tol=1
    chk=1
    huber = np.divide((1.345*m1/0.6745),np.abs(r))
    w = np.minimum(np.ones(huber.shape[0]),huber)
    
    # run now till convergence using iterated reweighted least squares
    while tol > 1e-6:
        
        # robust ridge regression
        W = np.diagflat(w)
        bhat = lin.inv(x.T @ W @ x + k*np.eye(x.shape[1])) @ (x.T @ W @ y)
        yhat = x@bhat
        r = y-yhat
        huber = np.divide((1.345*m1/0.6745),np.abs(r))
        w = np.minimum(np.ones(huber.shape[0]),huber)
        tol = lin.norm(g-w)
        g=w
        
        # regression diagnostics
        h = x @ (lin.inv(x.T @ W @ x + k*np.eye(x.shape[1])) @ (x.T @ W))
        edf = np.trace(h)
        r2 = (yhat[:,None].T @ W @yhat[:,None]) @ lin.inv(y[:,None].T @ W @y[:,None])
        
        # limit check on iterations
        chk = chk+1
        if chk==1500:
            break
        print(f'Tolerance is  {tol:.6f}')
    
    print(f'Converged in {chk} iterations')
    return bhat,edf,r,r2,w
    

##### CREATE DUMMY DATA #####
#20Hz sine sampled at 100Hz in noise
y = np.sin(2*np.pi*(20/100)*np.arange(100)) + rnd.randn(100,)
#1000 predictors with only 100 data points
x = rnd.randn(100,1000)
# predictor 20 is important
x[:,19] = y*7 + rnd.randn(100,)
# correlated predictor 200 is important
x[:,199] = x[:,19]*3.5 + rnd.randn(100,)
# introduce outliers in the data
y[19:24] = y[19:24] + 8
# ridge coefficient of 10
ridge_coeff=1

bhat,edf,r,r2,w = robust_ridge(x,y,ridge_coeff)
plt.stem(bhat)
plt.suptitle('Coeff values')
plt.xlabel('Predictors')
plt.ylabel('Magnitude')
    
    
    
    
    
    
        
    