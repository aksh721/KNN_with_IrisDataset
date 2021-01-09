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

# # Project : Apply Classification tech. of ML on iris Dataset.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("C:\\Users\\AMAN BIRADAR\\Downloads\\Python\\iris.csv")
df.head(5)


# In[3]:


# Step 4: Exploratory Data Analysis
df.info()


# In[4]:


df.describe()


# In[5]:


df['species'].nunique()


# In[6]:


df['species'].unique()


# In[7]:


sns.scatterplot(x=df['sepal_length'],y=df['petal_length'],hue=df['species'])


# In[8]:


# Step 5: Building a Machine Learning Model

X= df.iloc[:,:4]
y = df['species']


# In[9]:


X.head(5)


# In[10]:


y


# In[14]:


# Split Dataset into training and testing

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)


# In[15]:


print("X_train shape",X_train.shape)
print("X_test shape",X_test.shape)

print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
K_value = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]
accuracy_val = []
for i in K_value : 
    Knn_model = KNeighborsClassifier(n_neighbors=i)
    Knn_model.fit(X_train,y_train)
    y_pred = Knn_model.predict(X_test)
    acc = metrics.accuracy_score(y_pred,y_test)
    accuracy_val.append(acc)
print("All 15 Model trained Sucessfully...")


# In[17]:


K_value


# In[18]:


print(accuracy_val)


# In[19]:


plt.plot(K_value,accuracy_val,'-*')
plt.xlabel("K-Value")
plt.ylabel("Accuracy_val")


# In[20]:


from sklearn.neighbors import KNeighborsClassifier

Knn_model = KNeighborsClassifier(n_neighbors=9)
Knn_model.fit(X_train,y_train)
y_pred = Knn_model.predict(X_test)
acc = metrics.accuracy_score(y_pred,y_test)
print("Model accuracy for k=9 is : ",acc*100)


# In[22]:


SL = float(input("Enter SL : "))
SW = float(input("Enter SW : "))
PL = float(input("Enter PL : "))
PW = float(input("Enter PW : "))


# In[24]:


# 5.0	3.6	1.4	0.2	setosa
# Step 7: Predictions
# predict() : use to predict results.
result = Knn_model.predict([[SL,SW,PL,PW]])
print("Predicted result is : ", result[0])

