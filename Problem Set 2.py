#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Import the Libraries numpy and pandas 

import numpy as np
import pandas as pd

# Load the csv file 

df = pd.read_csv('BWGHT.csv')

# Create a linear model class
class linear_model:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.b = np.linalg.solve(x.T@x,x.T@y)
        e = y-x@self.b
        self.vb = self.vcov_b(e)
        self.se = np.sqrt(np.diagonal(self.vb))
        self.t = self.b/self.se
    def vcov_b(self,e):
        x = self.x
        return e.var()*np.linalg.inv(x.T@x)
    def create_meat(e):
        return np.diagflat(e.values**2)

class newey_west(linear_model):
    def create_meat(self,e):
        n = len(e)
        meat = np.zeros((n,n))
        var=0
        covar=0
        for i in range(0,n):
            var += e[i]**2
        for i in range(0,n-1):
            covar += e[i]*e[i+1]
        var=var/n
        covar=covar/(n-1)
        for i in range(0,n):
            meat[i,i]=var # All the diagonal elements 
            if i<n-1:
                meat[i,i+1]=covar  # All the off diagonal elements 
                meat[i+1,i]=covar
        return meat
    
    def vcov_b(self,e):
        x = self.x
        meat = self.create_meat(e)
        bread = np.linalg.inv(x.T@x)@x.T
        sandwich = bread@meat@bread.T
        # print(meat)  
        return sandwich
    
df['(intercept)'] = 1
x = df[['(intercept)','cigs','faminc']]
y = df['bwght']
# Print the NEWEY_WEST estimatot results for the above mention x and y
print('Newey west estimator results:',newey_west(x,y).t)







# In[25]:



# Now we can run the simulation to check if the newey_west algorithm has a better rejection ratio than the basic linear 
# estimator 
# Our linear model has a ratio of around 10.7% ( for a random seed)
# Newey west estimator has a ratio of 6.7% (for a random seed)which is better than the linear model.
n = 1000
nsim = 1000
reject = 0
for isim in range(nsim):
    x = np.random.normal(size=(n,1))
    y = np.random.normal(size=(n,))
    for i in range(1,n):
        x[i,:] += 0.5*x[i-1,:]
        y[i] += 0.5*y[i-1]
    x = np.hstack([np.ones([n,1]),x])
    if np.abs(newey_west(x,y).t[1])>1.96:
    #if np.abs(linear_model(x,y).t[1])>1.96:
        reject += 1
print('The rejection rate for the newey west is', reject/nsim)


# In[ ]:





# In[ ]:




