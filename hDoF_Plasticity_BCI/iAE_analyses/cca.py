import numpy as np
import scipy as sp
import scipy.stats as stats
import numpy.linalg as lin
import matplotlib.pyplot as plt
import math as math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_decomposition import CCA


def cca(Xa,Xb):
    
    # demean the data    
    for i in range(Xa.shape[1]):
        Xa[:,i] = Xa[:,i] - np.mean(Xa[:,i])
        
        
    for i in range(Xb.shape[1]):
        Xb[:,i] = Xb[:,i] - np.mean(Xb[:,i])
        
    # sample covariance matrices
    #Caa = np.cov(Xa,rowvar=False)
    #Cbb = np.cov(Xb,rowvar=False)
    Caa = (1/(Xa.shape[0]-1)) * (np.transpose(Xa) @ Xa)
    Cbb = (1/(Xb.shape[0]-1)) * (np.transpose(Xb) @ Xb)
    Cab = (1/(Xa.shape[0]-1)) * (np.transpose(Xa) @ Xb)
    
    # check for rank
    if lin.matrix_rank(Caa) < Caa.shape[0]:
        Caa = Caa + 1e-8 * np.identity(Caa.shape[0])
    
    # cholesky factorization
    Caa12 = (lin.cholesky(Caa)).T
    Cbb12 = lin.cholesky(Cbb).T
        
    # solver    
    X = (lin.inv(Caa12.T) @ Cab) @ (lin.inv(Cbb12))
    #X = lin.solve(Caa12.T, lin.solve(Cbb12,Cab))
    U,S,V = lin.svd(X,full_matrices=False)
    #U,S,V = sp.linalg.svd(X)
    Wa = lin.inv(Caa12) @ U;
    Wb = lin.inv(Cbb12) @ V;
    
    
    #canonical variates
    Za = Xa @ Wa;
    Zb = Xb @ Wb;
    
    # return the output
    return Wa,Wb,S,Za,Zb



Xa = np.random.randn(200,96)
Xb = np.random.randn(200,96)
Wa,Wb,S,Za,Zb = cca(Xa,Xb)
print(S)
#plt.plot(Za[:,0],Zb[:,0],'.')
plt.stem(S)
print(S[1:10])


cca_func = CCA(n_components=2)
cca_func.fit(Xa,Xb)


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
    






