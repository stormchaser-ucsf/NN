import numpy as np
import scipy as sp
import scipy.stats as stats
import numpy.linalg as la
import matplotlib.pyplot as plt
import math as math
from mpl_toolkits.mplot3d import Axes3D


def cca(Xa,Xb):
    
    # demean the data    
    for i in range(Xa.shape[1]):
        Xa[:,i] = Xa[:,i] - np.mean(Xa[:,i])
        
        
    for i in range(Xb.shape[1]):
        Xb[:,i] = Xb[:,i] - np.mean(Xb[:,i])
        
    # sample covariance matrices
    Caa = np.cov(Xa,rowvar=False)
    Cbb = np.cov(Xb,rowvar=False)
    Cab = (1/(Xa.shape[0]-1)) * (np.transpose(Xa) @ Xb)
    
    # check for rank
    if la.matrix_rank(Caa) < Caa.shape[0]:
        Caa = Caa + 1e-8 * np.identity(Caa.shape[0])
    
    # cholesky factorization
    Caa12 = la.cholesky(Caa)
    Cbb12 = la.cholesky(Cbb)
        
    # solver    
    X = la.inv(np.transpose(Caa12) ) @ Cab @ la.inv(Cbb12)
    U,S,V = la.svd(X)
    Wa = la.inv(Caa12) @ U;
    Wb = la.inv(Cbb12) @ V;
    
    #canonical variates
    Za = Xa @ Wa;
    Zb = Xb @ Wb;
    
    # return the output
    return Wa,Wb,S,Za,Zb



Xa = np.random.randn(20,2)
Xb = np.random.randn(20,7)
Wa,Wb,S,Za,Zb = cca(Xa,Xb)
print(Wa)
print(S)
print(Wb)
plt.plot(Za[:,0],Zb[:,0],'.')


#### some random class stuff

class Robot:
    def __init__(self, n,c,w):
        self.name = n
        self.color = c
        self.weight = w
        
    def introduce_self(self):
        print("My name is " + self.name)
        

        
r1 = Robot("Tom","white","30")
        
class Person:
    def __init__(self,n,h):
        self.name = n
        self.height = h
        
    def displayht(self):
        print("My ht is " + self.height)


x1= Person("niki","50")   
x1.robot = r1
x1.robot.introduce_self()

Robot.introduce_self(r1)
Person.displayht(x1)
    






