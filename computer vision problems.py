#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data()
X_train.shape


# In[3]:


def plot_sample(X,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    plt.xlabel(classes[y[index][0]])


# In[4]:


plot_sample(X_train,y_train,0)


# In[5]:


plot_sample(X_train,y_train,1)


# In[6]:


X_train=X_train/255
X_test=X_test/255


# In[7]:


ann=models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(3000,activation='relu'),
    layers.Dense(3000,activation='relu'),
    layers.Dense(10,activation='sigmoid')
])


# In[8]:


ann.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[9]:


ann.fit(X_train,y_train,epochs=5)


# In[10]:


ann.evaluate(X_test,y_test)

