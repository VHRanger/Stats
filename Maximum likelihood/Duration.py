import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy as sc
import statsmodels.tools.numdiff as smt
import numpy.random as rd

nn = 20000#sample size

#create random stuff
z1 = rd.rand(nn)
z2 = rd.rand(nn)
z1 = rd.rand(nn)

#create randomized attributes
x1 = 0.35*z1 + 0.65*z2
x2 = 0.65*z1 + 0.35*z2

#True parameters, const, x1, x2 and p
betas = np.array([-0.8, -0.5, 0.5, 2])

#lambda in model
lambd = np.exp(betas[0] + betas[1]*x1 + betas[2]*x2)

# create new random noise, then use it to create a related variable, w
u = rd.rand(nn)
z = -np.log(1-u)*lambd**(-1)
w = ((-np.log(1-u))**(1/betas[3]))/lambd ######problem was here

#censored version of w at 3
wcensor3 = np.array([])
for i in range(len(w)):
    if w[i] < 3:
        wcensor3 = np.append(wcensor3, w[i])
    else:
        wcensor3 = np.append(wcensor3, 3)
        
#Create df to compare OLS to MLE
df = pd.DataFrame(x1, columns=['x1'])
df = df.join(pd.DataFrame(x2, columns=['x2']))
df = df.join(pd.DataFrame(w, columns=['w']))
df = df.join(pd.DataFrame(wcensor3, columns=['wcensor3']))    

#estimate OLS for censored and normal models
OLSw = smf.ols('w ~ x1+x2', data = df).fit()
OLScensor = smf.ols('wcensor3 ~ x1+x2', data = df).fit()


#Now time to do MLE
#Initial Guesses
b_hats = np.array([0.9, 0.1, -0.3, 0.75])
p = b_hats[3] #renaming p for clarity
dur = wcensor3#same

#Doing it 3 ways, exp, Weibul and censored Weibul

def dur_exp(b_hats):
    count = 0
    for i in range(0, len(dur)):
        xb = b_hats[0]+b_hats[1]*x1[i]+b_hats[2]*x2[i]
        lam = np.exp(xb)
        llfz = xb - lam*z[i]
        count += llfz    
    return -count
    
def dur_Weibul(b_hats):
    count = 0
    for i in range(len(dur)):
        xb = b_hats[0]+b_hats[1]*x1[i]+b_hats[2]*x2[i]
        lam = np.exp(xb)        
        llfw3 = np.log(p) + p*xb + (p-1)*np.log(w[i]) - (lam*w[i])**p
        count += llfw3
    return -count

def dur_censored(b_hats):
    p = b_hats[3]
    count = 0
    censor = lambda x: 1 if (dur[x] <3) else 0
    #create array of 1 and 0 to create dual ll functions later on                 
    for i in range(len(dur)):
        xb = b_hats[0]+b_hats[1]*x1[i]+b_hats[2]*x2[i]
        lam = np.exp(xb)
        lldur = censor(i)*(np.log(p) + xb + (p-1)*np.log(lam*dur[i]) - (lam*dur[i])**p) - (1-censor(i))*((3*lam)**p)
        count += lldur
    return -count
    
def durcens(betas):
    p = betas[3]
    count =0
    for i in range(len(dur)):
        xb = betas[0] + betas[1]*x1[i] + betas[2]*x2[i]
        lamb= np.exp(xb)
        if dur[i]<3:
            count+= dur[i]*np.log(p)+(p*xb) + (p-1)*np.log(dur[i])-(lamb*dur[i])**p
        else: 
            count+= (p*xb) + (p-1)*np.log(dur[i])-(lamb*dur[i])**p - ((dur[i])*(3*lamb)**p)
    return(-count)



#exponential distribution on non-censored model
b_exp = b_hats[0:3]
EXPest = sc.optimize.minimize(dur_exp, b_exp, method='nelder-mead')

#Weibull model on censored distribution
WBest = sc.optimize.minimize(dur_Weibul, b_hats, method='nelder-mead')
#Censored model
CENSest = sc.optimize.minimize(dur_censored, b_hats, method='nelder-mead')

#Calculate CRLB for MLE from inverse Hessian


print betas
print OLSw.summary()
print OLScensor.summary()

print EXPest
print WBest
print CENSest
