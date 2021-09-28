#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Import the libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Simulation for different Sample Sizes : 
def samplesim(nsim1,rho1,alpha1):
    nvec = [50,100,250,500,1000]
    outpn = np.zeros([nsim,len(nvec)])
    biasn = np.zeros([nsim,len(nvec)])
    for ns in range(len(nvec)):
        n = nvec[ns]
        for isim in range(nsim):
            e = np.random.normal(size=(n,))
            x = e.copy()
            for i in range(1,n):
                x[i]=alpha1+rho*x[i-1]+e[i]
            ymat = x[1:]
            xmat = x[:-1].reshape(-1,1)
            xmat = np.hstack([np.ones((n-1,1)),xmat])
            b = np.linalg.solve(xmat.T@xmat,xmat.T@ymat)
            outpn[isim,ns] = b[1]
            biasn[isim,ns] = rho-outpn[isim,ns]
    RSS = np.zeros([len(nvec),])
    for ns in range(len(nvec)):
        for i in range(nsim):
            RSS[ns,] += (biasn[i,ns])**2
    print("Residual sum of squares for different sample size",RSS)
    print("Bias for different sample sizes ",biasn.mean(0))
    plt.figure()
    for ns in range(len(nvec)):
        sns.kdeplot(biasn[:,ns],label=nvec[ns])

#Simulation for different Aplha value in the AR model 
def alphasim(nsim1,rho1,samplesize):
    avec = [-5,0,5,10,15]
    outp = np.zeros([nsim,len(avec)])
    bias = np.zeros([nsim,len(avec)])
    for ns in range(len(avec)):
        alpha = avec[ns]
        for isim in range(nsim):
            e = np.random.normal(size=(n,))
            x = e.copy()
            for i in range(1,n):
                x[i]=alpha+rho*x[i-1]+e[i]
            ymat = x[1:]
            xmat = x[:-1].reshape(-1,1)
            xmat = np.hstack([np.ones((n-1,1)),xmat])
            b = np.linalg.solve(xmat.T@xmat,xmat.T@ymat)
            outp[isim,ns] = b[1]
            bias[isim,ns] = rho-outp[isim,ns]
    RSS = np.zeros([len(avec),])
    for ns in range(len(avec)):
        for i in range(nsim):
            RSS[ns,] += (bias[i,ns])**2
    print("Residual sum of squares for different alpha values",RSS)
    print("Bias for different alpha values ",bias.mean(0))
    plt.figure()
    for ns in range(len(avec)):
        sns.kdeplot(bias[:,ns],label=avec[ns])

#Simulation for different rho Values in the AR model 
def rhosim(nsim1,samplesize,alpha1):
    rvec = [0.1,0.2,0.4,0.6,0.8]
    outp = np.zeros([nsim,len(rvec)])
    bias = np.zeros([nsim,len(rvec)])
    for ns in range(len(rvec)):
        rho = rvec[ns]
        for isim in range(nsim):
            e = np.random.normal(size=(n,))
            x = e.copy()
            for i in range(1,n):
                x[i]=alpha1+rho*x[i-1]+e[i]
            ymat = x[1:]
            xmat = x[:-1].reshape(-1,1)
            xmat = np.hstack([np.ones((n-1,1)),xmat])
            b = np.linalg.solve(xmat.T@xmat,xmat.T@ymat)
            outp[isim,ns] = b[1]
            bias[isim,ns] = rho-outp[isim,ns]
    RSS = np.zeros([len(rvec),])
    for ns in range(len(rvec)):
        for i in range(nsim):
            RSS[ns,] += (bias[i,ns])**2
    print("Residual sum of squares for different rho values",RSS)
    print("Bias for different rho values ",bias.mean(0))
    plt.figure()
    for ns in range(len(rvec)):
        sns.kdeplot(bias[:,ns],label=rvec[ns])

# In[8]:


# Lets input few values
nsim = 10000
rho = 0.4
n = 500
alpha_value = 1


# In[9]:


# Print the simlation for different ALPHA values : 
alphasim(nsim,rho,n)


# In[10]:


# Print the simulation for different Sample sizes :
samplesim(nsim,rho,alpha_value)


# In[11]:


# Print the simulation for different Rho values : 
rhosim(nsim,n,alpha_value)


# In[5]:





# In[ ]:




