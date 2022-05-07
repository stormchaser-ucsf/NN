# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:14:04 2022

@author: Nikhlesh
"""


import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import matplotlib.pyplot as plt
import math as math
from mpl_toolkits.mplot3d import Axes3D

# create the first wave
list1 = range(200)
x = [math.sin(2*math.pi*1/100*i) for i in list1] # sin
plt.plot(x)
    



# create the second wave
n = 0.5*randn(size(x))
y = [i*1.5 for i in x] + n
plt.plot(y)


# plot them both against each other 
figure = plt
plt.plot(x,y,'.')


### matlab code
#corr_xy = (x'*y)/(norm_x*norm_y) %cos_theta
#acosd(corr_xy)


##python equivalent
corr_xy = np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
print(corr_xy)




