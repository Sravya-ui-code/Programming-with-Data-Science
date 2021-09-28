#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

#Hierarchial Cluserting 
#Import necessary modules
import numpy as np
import matplotlib.pyplot as plt

#create sample data
X=np.array([
        [0.4,0.53],
        [0.22,0.32],
        [0.35,0.32],
        [0.26,0.19],
        [0.08,0.41],
        [0.35,0.3],
        [0.8,0.98],
        [0.28,0.33]
])

class Distance_computation_grid(object):
    def __init__(self):
        pass
    
    def compute_distance(self,samples):
        distance_mat=np.zeros((len(samples),len(samples)))
        for i in range(distance_mat.shape[0]):
            for j in range(distance_mat.shape[0]):
                if i==j:
                    distance_mat[i,j] = 10**4
                else:
                    distance_mat[i,j] = float(self.calculate_distance(samples[i],samples[j]))
        return distance_mat
    
    def calculate_distance(self,samplei,samplej):
        dist=[]
        for i in range(len(samplei)):
            for j in range(len(samplej)):
                try:
                    dist.append(np.linalg.norm(np.array(samplei[i])-np.array(samplej[j])))
                except:
                    dist.append(self.intersampledist(samplei[i],samplej[j]))
        return min(dist)
    
    def intersampledist(self,s1,s2):
        if str(type(s2[0]))!='<class \'list\'>':
            s2=[s2]
        if str(type(s1[0]))!='<class \'list\'>':
            s1=[s1]
        m = len(s1)
        n = len(s2)
        dist = []
        if n>=m:
            for i in range(n):
                for j in range(m):
                    if (len(s2[i])>=len(s1[j])) and str(type(s2[i][0])!='<class \'list\'>'):
                        dist.append(self.interclusterdist(s2[i],s1[j]))
                    else:
                        dist.append(np.linalg.norm(np.array(s2[i])-np.array(s1[j])))
        else:
            for i in range(m):
                for j in range(n):
                    if (len(s1[i])>=len(s2[j])) and str(type(s1[i][0])!='<class \'list\'>'):
                        dist.append(self.interclusterdist(s1[i],s2[j]))
                    else:
                        dist.append(np.linalg.norm(np.array(s1[i])-np.array(s2[j])))
        return min(dist)
    
    def interclusterdist(self,cl,sample):
        if sample[0]!='<class \'list\'>':
            sample = [sample]
        dist   = []
        for i in range(len(cl)):
            for j in range(len(sample)):
                dist.append(np.linalg.norm(np.array(cl[i])-np.array(sample[j])))
        return min(dist)



# In[7]:


progression = [[i] for i in range(X.shape[0])]
samples     = [[list(X[i])] for i in range(X.shape[0])]
m=len(samples)
distcal = Distance_computation_grid()




# In[8]:


while m>1:
    print('Sample size before clustering    :- ',m)
    Distance_mat      = distcal.compute_distance(samples)
    sample_ind_needed = np.where(Distance_mat==Distance_mat.min())[0]
    value_to_add      = samples.pop(sample_ind_needed[1])
    samples[sample_ind_needed[0]].append(value_to_add)
    
    print('Cluster Node 1                   :-',progression[sample_ind_needed[0]])
    print('Cluster Node 2                   :-',progression[sample_ind_needed[1]])
    
    progression[sample_ind_needed[0]].append(progression[sample_ind_needed[1]])
    progression[sample_ind_needed[0]] = [progression[sample_ind_needed[0]]]
    v = progression.pop(sample_ind_needed[1])
    m = len(samples)
    
    print('Progression(Current Sample)      :-',progression)
    print('Cluster attained                 :-',progression[sample_ind_needed[0]])
    print('Sample size after clustering     :-',m)
    print('\n')



# In[9]:


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
Z = linkage(X, 'single')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)








