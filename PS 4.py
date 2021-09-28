#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[4]:
# Import Libraries : 

import numpy as np
import pandas as pd

# Beta Estimator : 

class Ridge():
    def __init__(self,x,y,lamda):
        self.x=x
        self.y=y
        self.lamda=lamda
        k=x.T.dot(x) + lamda*np.eye(x.shape[1])
        self.beta = np.linalg.inv(k).dot(x.T.dot(y))  # Equation of the beta ridge 
        y_pred=x.dot(self.beta)
    def ridge_params(self):
        return self.beta
    def rmse(self):
        y_pred=x.dot(self.beta)
        mse=((y-y_pred)**2).mean()
        rmse = np.sqrt(mse)
        return rmse

x=np.random.randn(150,50)
y=np.random.randn(150)
ridge = Ridge(x,y,5)
print(ridge.ridge_params())
print('RMSE is',ridge.rmse())

#Checking with sk-learn linear model Ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ridge = Ridge(alpha=5)
ridge.fit(x,y)
print(ridge.coef_)
y_pred=ridge.predict(x)
rmse = np.sqrt(mean_squared_error(y,y_pred))
print('RMSE is',rmse)





