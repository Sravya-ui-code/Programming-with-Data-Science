#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:

def sequence(x,lent):
    seq_elem = [x]
    for i in range(1,2*lent):
        if x%2 == 0: # When x is even 
            seq_elem.append(x//2)
            x = x//2
        else : # When x is odd 
            seq_elem.append(a*x+b)
            x = a*x+b
    print("The sequence for a =",a,"b =",b,'is :',seq_elem)
    for i in range(1, len(seq_elem)):
        if seq_elem[0] != seq_elem[i]:
              continue
        else:
            print('The sequence converges for a=',a,'b=',b, 'The above sequence is a holding pattern') # Displays if the holding pattern converges 
            break
        
def generalized_hailstone(): # Generalized solution for a and b in range(0,11)
    global a # Value of a 
    global b # Value of b 
    global lent # Length of the sequence 
    x = int(input("Enter the starting number:")) # Starting number  
    lent = int(input("Enter the sequence length:")) # If lent = x, sequence is generated from (1,2*x)
    for a in range(1,11):
        for b in range(1,11):
            sequence(x,lent)
generalized_hailstone()


# In[ ]:





# In[ ]:




