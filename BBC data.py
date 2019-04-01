
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[8]:


import sklearn


# In[9]:


df = pd.read_csv("bbc.csv")


# In[10]:


df


# In[11]:


array = df.values


# In[17]:


X = df.values[:,0:11] 
Y = array[:,11]


# In[22]:


df.groupby("BikeBuyer").size()


# In[23]:


from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC


# In[29]:


model = SVC()


# In[75]:


test_size = 0.3


# In[76]:


seed = 1


# In[77]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[78]:


model.fit(X_train, Y_train)


# In[79]:


result = model.score(X_test, Y_test)
result


# In[80]:


Accuracy = result * 100


# In[81]:


Accuracy


# In[86]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[87]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[88]:


model.fit(X_train, Y_train)


# In[89]:


result = model.score(X_test, Y_test)
result


# In[91]:


Accuracy = result * 100
Accuracy


# In[94]:


from sklearn.ensemble import GradientBoostingClassifier


# In[95]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[96]:


model = GradientBoostingClassifier()
model.fit(X_train, Y_train)


# In[103]:


result = model.score(X_test, Y_test)
result


# In[105]:


Accuracy = result*100
Accuracy


# In[107]:


test_size = .30
seed = 45
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = SVC()
model.fit(X_train, Y_train) 

result = model.score(X_test, Y_test)

result*100

