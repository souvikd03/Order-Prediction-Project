#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install lightgbm


# # Step-1: Import dataset

# In[2]:


#import libs
import pandas as pd
import numpy as np


# In[3]:


dataset = pd.read_csv("supplement.csv")
dataset


# # Step-2: Data Inspection and data handling

# In[4]:


dataset.info()


# In[5]:


dataset.isnull().sum() 


# # Step-3: Data Visualization

# In[6]:


#!pip install plotly==5.11.0
import plotly.express as px 


# In[7]:


pie1 = dataset["Store_Type"].value_counts()
store = pie1.index
orders = pie1.values

fig = px.pie(dataset,values=orders,names=store)
fig.show()


# In[8]:


pie2 = dataset["Location_Type"].value_counts()
location = pie2.index
orders = pie2.values

fig = px.pie(dataset,values=orders,names=location)
fig.show()


# In[9]:


pie1 = dataset["Holiday"].value_counts()
Holiday = pie1.index
orders = pie1.values

fig = px.pie(dataset,values=orders,names=Holiday)
fig.show()


# In[10]:


pie1 = dataset["Discount"].value_counts()
Discount = pie1.index
orders = pie1.values

fig = px.pie(dataset,values=orders,names=Discount)
fig.show()


# # Step-4: Data Preprocessing

# In[11]:


dataset["Discount"] = dataset["Discount"].map({"No": 0,"Yes": 1})


# In[12]:


dataset


# In[13]:


dataset["Store_type"] = dataset["Store_type"].map({"S1": 1,"S2": 2,"S3": 3,"S4": 4})


# In[14]:


dataset


# In[15]:


dataset["Location_type"] = dataset["Location_type"].map({"L1": 1,"L2": 2,"L3": 3,"L4": 4,"L5": 5})


# In[16]:


dataset


# In[17]:


X = np.array(dataset[["Store_type", "Location_type", "Holiday", "Discount"]])
y=np.array(dataset["Order"])


# In[18]:


X


# In[19]:


y


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=10)


# In[22]:


len(X_train)


# # Step-5: Building the ML Model

# In[25]:


#pip install lightgbm
import lightgbm as ltb
model = ltb.LGBMRegressor()


# In[26]:


model.fit(X_train,y_train)


# In[27]:


y_pred = model.predict(X_test)


# In[28]:


y_pred


# In[29]:


y_test


# In[30]:


data = pd.DataFrame(data={"Predicted Orders": y_pred.flatten()})


# In[31]:


data
# Order-Prediction-Project
