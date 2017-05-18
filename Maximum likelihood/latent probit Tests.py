###Probit Model###
"""
Note there is namespace conflict between modules.
You are supposed to use the modules individually after
generating the initial data
"""

from numpy import random as rd
import numpy as np
#To minimize log likelihood functions
from scipy.optimize import minimize
#tools around normal distributions
from scipy.stats import norm
import scipy as sc
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf #to validate probit
from statsmodels.base.model import GenericLikelihoodModel #StatsM method
import statsmodels.tools.numdiff as smt
import matplotlib.pyplot as plt
%matplotlib inline

######################
#FAKE DATA
######################


#sample size
n = 10000

#generators
z1 = rd.randn(n)
z2 = rd.randn(n)
l1 = np.clip(z1, -0.000001, 0.000001)
l2 = np.clip(z2, -0.000001, 0.000001)
l1[l1>0] = 1
l1[l1<0] = 0
l2[l2>0] = 1
l2[l2<0] = 0

#create artificial parameters
x1 = 0.8*z1 + 0.2*z2
x2 = 0.2*z1 + 0.8*z2
u = 2*rd.randn(n)

#create LHS variable from above
ystar = 0.5 + 0.75*l1 - 0.75*l2 + u 

#Matrix of endo variables with prepended intercept vector
X = sm.add_constant(np.column_stack((l1, l2)))

#create artificial observed data in [1,2,3,4,5] range
y = np.array([]) 
def create_dummy(array, data, cutoff_low,c_m1, c_m2, cutoff_hi):
    for i in range(0, len(data)):
        if data[i] <= cutoff_low:
            array = np.append(array, 1)
        elif data[i] > cutoff_low and data[i] <= c_m1:
            array = np.append(array, 2)
        elif data[i] > c_m1 and data[i] <= c_m2:
            array = np.append(array, 3)
        elif data[i] > c_m2 and data[i] <= cutoff_hi:
            array = np.append(array, 4)
        else:
            array = np.append(array, 5)
    return array
       
y = create_dummy(y, ystar, -1, -0.5, 0, 0.5)

#Initial MLE Guesses for both betas and the two cutoff points
bhats = np.array([0.1,0.2,0.3,0.4,0.5,0.6])

#####################################
#ORDERED PROBABILITY MODEL
#####################################

def OrderProbit(betas, data):#very slow pure python version
    result = 0
    selection = lambda x,s: 1 if (y[x] == s) else 0
    for i in range(0, len(data)):
        xb = betas[4]*x1[i] + betas[5]*x2[i]
        prob1 = norm.cdf(betas[0] - xb)
        prob2 = norm.cdf(betas[1] - xb) - norm.cdf(betas[0] - xb)
        prob3 = norm.cdf(betas[2] - xb) - norm.cdf(betas[1] - xb)
        prob4 = norm.cdf(betas[3] - xb) - norm.cdf(betas[2] - xb)
        prob5 = 1 - norm.cdf(betas[3] - xb)
        llf = (selection(i,1)*np.log(prob1) + selection(i,2)*np.log(prob2) + 
            selection(i,3)*np.log(prob3) + selection(i,4)*np.log(prob4) + 
            selection(i,5)*np.log(prob5))
        result += llf
    return -result

def OrderProbit5(betas, X, y): #faster version of above
   xb1 = np.dot(X, np.append(betas[0], betas[4:]))
   xb2 = np.dot(X, np.append(betas[1], betas[4:])) 
   xb3 = np.dot(X, np.append(betas[2], betas[4:]))
   xb4 = np.dot(X, np.append(betas[3], betas[4:]))   
   llf = np.sum(np.log(
           ((y==1) * (norm.cdf(xb1)))  +        
           ((y==2) * (norm.cdf(xb2) - norm.cdf(xb1)))  +
           ((y==3) * (norm.cdf(xb3) - norm.cdf(xb2)))  +
           ((y==4) * (norm.cdf(xb4) - norm.cdf(xb3)))  +
           ((y==5) * (1-norm.cdf(xb4)))
           ))
   return -llf
   
def OrderProbitRedux(betas): #same as above rewrote
   c1 = betas[0]
   c2 = betas[1]
   c3 = betas[2]
   c4 = betas[3]
   xb = np.dot(X, np.append(betas[4], betas[5:])) 
   llf = np.sum(np.log(
           ((y==1) * (norm.cdf(c1 - xb)))  +        
           ((y==2) * (norm.cdf(c2 - xb) - norm.cdf(c1 - xb)))  +
           ((y==3) * (norm.cdf(c3 - xb) - norm.cdf(c2 - xb)))  +
           ((y==4) * (norm.cdf(c4 - xb) - norm.cdf(c3- xb)))  +
           ((y==5) * (1-norm.cdf(c4- xb)))
           ))
   return -llf
   
   
probit_est = minimize(OrderProbit5, bhats, method='nelder-mead', options={'maxiter':500000, 'maxfev':500000})
    

############################
#LATENT DATA
###########################
X = sm.add_constant(np.column_stack((l1, l2)))

rng = rd.rand(n)
p = np.array([])
for i in range(n):
    if rng[i] > 0.5:
        p = np.append(p, 1)
    else:
        p = np.append(p, 0)

ystar = np.array([])
for i in range(n):
    ystar = np.append(ystar, 
        0.5 + 
        p[i]*2*(0.75*l1[i] - 0.75*l2[i]) + 
        (1-p[i])*0.5*(0.5 + 0.75*l1[i] - 0.75*l2[i]) + u[i] )

y = np.array([]) 
def create_dummy(array, data, cutoff_low,c_m1, c_m2, cutoff_hi):
    for i in range(0, len(data)):
        if data[i] <= cutoff_low:
            array = np.append(array, 1)
        elif data[i] > cutoff_low and data[i] <= c_m1:
            array = np.append(array, 2)
        elif data[i] > c_m1 and data[i] <= c_m2:
            array = np.append(array, 3)
        elif data[i] > c_m2 and data[i] <= cutoff_hi:
            array = np.append(array, 4)
        else:
            array = np.append(array, 5)
    return array
       
y = create_dummy(y, ystar, -1.75, -0.5, 0.5, 1.75)        

bhats = np.array([0.1,0.2,0.3,0.4,0.5,0.6, 0.7])


############################
#LATENT ORDERED PROBIT
###########################
   
def LatentProbit25(betas): #faster version of above
    #Class probabilities are first element of each array
    #parameters for each class are following elements; 4 intercepts, and params
   p = np.sin(betas[0])**2
   xb11 = np.dot(X, np.append(betas[1], betas[5:])) #first class
   xb21 = np.dot(X, np.append(betas[2], betas[5:])) 
   xb31 = np.dot(X, np.append(betas[3], betas[5:]))
   xb41 = np.dot(X, np.append(betas[4], betas[5:]))
   xb12 = np.dot(X, np.append(betas[1], betas[5:])) #2nd class
   xb22 = np.dot(X, np.append(betas[2], betas[5:])) 
   xb32 = np.dot(X, np.append(betas[3], betas[5:]))
   xb42 = np.dot(X, np.append(betas[4], betas[5:]))
   llf = np.sum(np.log(
            p*np.exp(
           ((y==1) * (norm.cdf(xb11)))  +        
           ((y==2) * (norm.cdf(xb21) - norm.cdf(xb21)))  +
           ((y==3) * (norm.cdf(xb31) - norm.cdf(xb21)))  +
           ((y==4) * (norm.cdf(xb41) - norm.cdf(xb31)))  +
           ((y==5) * (1-norm.cdf(xb41)))
           )+
           (1-p)*np.exp(
           ((y==1) * (norm.cdf(xb12)))  +        
           ((y==2) * (norm.cdf(xb22) - norm.cdf(xb22)))  +
           ((y==3) * (norm.cdf(xb32) - norm.cdf(xb22)))  +
           ((y==4) * (norm.cdf(xb42) - norm.cdf(xb32)))  +
           ((y==5) * (1-norm.cdf(xb42)))
           )))
   return -llf
  
def LatentOrderProbit(betas): #WORK IN PROGRESS
   c1 = betas[0]
   c2 = betas[1]
   c3 = betas[2]
   c4 = betas[3]
   xb = np.dot(X, np.append(betas[4], betas[5:])) 
   llf = np.sum(np.log(
           ((y==1) * (norm.cdf(c1 - xb)))  +        
           ((y==2) * (norm.cdf(c2 - xb) - norm.cdf(c1 - xb)))  +
           ((y==3) * (norm.cdf(c3 - xb) - norm.cdf(c2 - xb)))  +
           ((y==4) * (norm.cdf(c4 - xb) - norm.cdf(c3- xb)))  +
           ((y==5) * (1-norm.cdf(c4- xb)))
           ))
   return -llf
  
probit_est = minimize(LatentProbit25, bhats, method='nelder-mead')





