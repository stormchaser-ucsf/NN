# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:30:40 2023

@author: nikic
"""


import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
from iAE_utils_models import *
import pandas as pd
from scipy import signal

df = pd.read_csv('monsoon_final.csv')
time=df['Time (s)']
I = df['Main Avg Current (mA)']
V = df['Main Avg Voltage (V)']

I = stats.zscore(I)
plt.figure()
plt.plot(time,I)

y=signal.savgol_filter(I,53,3)[:,None]
plt.plot(time,y)
plt.show()


# using a Kalman Filter to get a running estimate of the mean of the time series
xhat=[]
xhat.append(0)
p0 = 1
sigma_v = np.var(I[:round(1e4)])
for i in np.arange(1,len(I)):
    K = p0/(i*p0 + sigma_v)
    tmp = xhat[i-1] 
    


#%% using a Kalman Filter 
# estimate the mean shifts of the background sinusoid like signal
# can then use this to estimate shart, stop etc. using autocorrelation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy import signal


df = pd.read_csv('monsoon_final.csv')
time=df['Time (s)']
I = df['Main Avg Current (mA)']
V = df['Main Avg Voltage (V)']

# Initialize Kalman filter parameters to track the mean
x_est = 0 # State estimate (initial guess)
P_est = 1 # Error covariance matrix (initial guess)
F = 1 # State transition matrix
H = 1 # Measurement matrix
R = noise_var # Measurement noise variance

# VARY THESE TWO TO TRACK THE MEAN BEST
Q = 1e-6 # Process noise covariance matrix (TUNING PARAMETER)
noise_var = 1e-3 #(TUNING PARAMETER)

# Run the Kalman filter
t=time
x_kf = np.zeros_like(t)
y_meas = I
for i in range(len(t)):
    # Predict the state and error covariance
    x_pred = F*x_est
    P_pred = F*P_est*F + Q
    
    # Update the state and error covariance based on the measurement
    K = P_pred*H/(H*P_pred*H + R)
    x_est = x_pred + K*(y_meas[i] - H*x_pred)
    P_est = (1 - K*H)*P_pred    
    
    # Save the estimated signal
    x_kf[i] = x_est

x_kf = stats.zscore(x_kf)
I = stats.zscore(I)

# low pass filter the data
Fs=round(5e3) #5khz
cutoff = 30
b,a = signal.butter(3,cutoff,fs=Fs,btype='low',analog=False) 
x_kf = signal.filtfilt(b, a, x_kf)

plt.figure()
plt.plot(t,I)
plt.plot(t,x_kf)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

# Define the true signal parameters
jump_prob = 0.01 # Probability of a jump at each time step
jump_var = 0.5 # Variance of the Gaussian jump
x_true = 1 # True constant

# Generate the true signal
t = np.linspace(0, 10, 101) # Time vector
x_true_vec = x_true*np.ones_like(t) # True constant signal
for i in range(1, len(t)):
    if np.random.rand() < jump_prob:
        x_true += np.sqrt(jump_var)*np.random.randn()
    x_true_vec[i] = x_true

# Add noise to the true signal
noise_var = 0.05 # Variance of the Gaussian noise
y_meas = x_true_vec + np.sqrt(noise_var)*np.random.randn(len(t)) # Measurement signal

# Initialize Kalman filter parameters
x_est = 0 # State estimate (initial guess)
P_est = 1 # Error covariance matrix (initial guess)

# Define the system matrices for Kalman filter
dt = t[1] - t[0] # Sampling interval
F = 1 # State transition matrix
Q = 1e-6 # Process noise covariance matrix (tuning parameter)
H = 1 # Measurement matrix
R = noise_var # Measurement noise variance

# Run the Kalman filter
x_kf = np.zeros_like(t)
Khist=[]
for i in range(len(t)):
    # Predict the state and error covariance
    x_pred = F*x_est
    P_pred = F*P_est*F + Q
    
    # Update the state and error covariance based on the measurement
    K = P_pred*H/(H*P_pred*H + R)
    x_est = x_pred + K*(y_meas[i] - H*x_pred)
    P_est = (1 - K*H)*P_pred
    Khist.append(K)
    
    # Save the estimated signal
    x_kf[i] = x_est


plt.figure()
plt.plot(t,I)
plt.plot(t,x_kf)
plt.show()

# Plot the true signal, measurement signal, and estimated signal
plt.plot(t, x_true_vec, '-b', t, y_meas, '.r', t, x_kf, '-g')
plt.legend(['True signal', 'Measurement signal', 'Estimated signal'])
plt.xlabel('Time')
plt.ylabel('Constant')
plt.show()

    