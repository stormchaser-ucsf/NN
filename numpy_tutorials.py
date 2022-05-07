# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:16:10 2022

@author: Nikhlesh
"""

import numpy as np
import numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt

# creates an array of size 5,2 rows by columns
a = np.arange(10).reshape(5, 2)
# want to access only the last row of a
a[-1,:]
# want to access the last two rows of a
a[-2:]
# want to access the last three rows of a
a[-3:]
# accessing rows 2 and beyond
a[1:]
# want to acesss the first column of a
a[:,0]
# want to access the second column of a
a[:,1]
# dimension of a
a.ndim
# data tpe
a.dtype
# size
a.size
# create an array increasing numbers
b=np.arange(0,2,0.1)
# can use linspace here same as matlab
x=np.linspace(0,1,100) #100 numbers betwee 0 and 1
#3d array
c = np.arange(24).reshape(2, 3, 4)  # 3d array
print(c)
# create a simple array and square the elements within in
x=np.arange(5)
x=x**2
print(x)
# now take the sin of this function
x=np.sin(x)
print(x)
# perform matrix multiplication
A = np.arange(10).reshape(5,2)
B = np.arange(10).reshape(2,5)
print(A)
print(B)
C= B @ A
print(C)
# some random number stuff
a=rnd.randn(3)
plt.hist(a)
print(a)
# manipulating this array
a+=3 # adding 3 to all the numbers
print(a)
a/=3 #diving it all by 3
print(a)
# more manipulation of arrays
a=np.arange(10).reshape(2,5)
print(a)
print(a.sum(axis=0))
print(a.max(axis=1))


# more fun stuff
a=np.arange(10)
print(a)
for i in a:
    print((i+1)*2)
a[2:5]
a[:10:3]=1000 #from 0 to 10, get every 3rd element
print(a)

# define funcitons
def f(x,y):
    return 10*x+y

c=f(10,3)
print(c)

# -1 in an array always implies the last row or column or whatver
b=np.arange(10).reshape(2,5)
print(b)
print(b[1,:])
for i in b.flat:
    print( i)

#flatten the arrau
c=np.ravel(b)
c=b;
# reshape the array
b.resize(5,2)
c=b
print(c)    


# some matplotlb plotting functions 
alpha = 0.7
phi_ext = 2 * np.pi * 0.5

def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)

phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T


fig = plt.figure(figsize=(14,6))

# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')

p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)

# surface_plot with color grading and color bar
ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)