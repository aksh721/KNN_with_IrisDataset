#!/usr/bin/env python
# coding: utf-8

# Step 1: Define the objective of the Problem Statement : From the given data try to classify the iris flower subspecies using SL,SW,PL and PW. 
# 
# Step 2: Data Gathering : Internet 
# 
# Step 3: Data Preparation : Not required bezcause already clean data.
# 
# Step 4: Exploratory Data Analysis 
# 
# Step 5: Building a Machine Learning Model
# 
# Step 6: Model Evaluation & Optimization 
# 
# Step 7: Predictions
# 

# # Project : Apply Classification tech. of ML on iris Dataset

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


df=pd.read_csv("C:\\Users\\AMAN BIRADAR\\Downloads\\Python\\iris.csv")
df.head()


# In[33]:


# Step 4: Exploratory Data Analysis
df.info()


# In[34]:


df.describe()


# In[35]:


df['species'].unique()


# In[36]:


df['species'].nunique()


# In[37]:


sns.scatterplot(x=df['sepal_length'],y= df['petal_length'],hue=df['species'])


# In[38]:


# Step 5: Building a Machine Learning Model

X = df.iloc[:,:4]
y = df['species']


# In[39]:


X.head(5)


# In[40]:


y


# In[41]:


# Split Dataset into training and testing

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.3,random_state = 3)


# In[42]:


print("X_train shape",X_train.shape)
print("X_test shape",X_test.shape)

print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)


# In[43]:


from sklearn.neighbors import KNeighborsClassifier

Knn_model = KNeighborsClassifier(n_neighbors=3)

# fit() : Use for model training.
Knn_model.fit(X_train,y_train)

print("KNN model  train sucessfully...!!!")


# In[44]:


SL = float(input("Enter SL : "))

SW = float(input("Enter SW : "))
PL = float(input("Enter PL : "))
PW = float(input("Enter PW : "))


# In[46]:


# 5.0	3.6	1.4	0.2	setosa
# Step 7: Predictions
# predict() : use to predict results.
result = Knn_model.predict([[SL,SW,PL,PW]])
print("Predicted result is : ", result[0])


# In[24]:


y_pred = Knn_model.predict(X_test)
y_pred


# In[25]:


from sklearn import metrics

acc = metrics.accuracy_score(y_pred,y_test)

print("Model Accuracy is : ",acc*100)


# In[26]:


#Second Model
from sklearn.neighbors import KNeighborsClassifier

Knn_model = KNeighborsClassifier(n_neighbors=5)

# fit() : Use for model training.
Knn_model.fit(X_train,y_train)

print("KNN model  train sucessfully...!!!")


# In[47]:


y_pred = Knn_model.predict(X_test)
y_pred


# In[48]:


from sklearn import metrics

acc = metrics.accuracy_score(y_pred,y_test)

print("Model Accuracy is : ",acc*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




