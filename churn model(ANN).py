#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd#to read csv
import keras#to create ANN


# In[2]:


df=pd.read_csv('bank.csv')


# In[3]:


df


# In[4]:


X=df.iloc[:, 3:13]
Y= df.iloc[:, 13]


# In[5]:


#creating dummies variables
geo=pd.get_dummies(X["Geography"], drop_first=True)
gen=pd.get_dummies(X['Gender'],drop_first=True)


# In[6]:


#concate the data frame
X=pd.concat([X,geo,gen], axis=1)


# In[7]:


#drop gender and geography columns
X=X.drop(['Geography', 'Gender'],axis=1)


# In[8]:


#splitting the dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size= 0.2, random_state=0)


# In[9]:


#feature Scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[10]:


#make Artificial neutral network
from keras.models import Sequential#for model
from keras.layers import Dense#to add preceptrons
from keras.layers import LeakyReLU, PReLU, ELU#for detemining the activation
from keras.layers import Dropout


# In[11]:


#initialising the Neutral network
model= Sequential()
#adding the imput layer and the first hidden layer
model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))


# In[12]:


#second layer
model.add(Dense(units=6, kernel_initializer= 'he_uniform', activation='relu'))
#output layer
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))


# In[13]:


#compiling the ANN
model.compile(optimizer = 'Adamax', loss= 'binary_crossentropy', metrics=['accuracy'])


# In[14]:


model_history= model.fit(X_train, Y_train, validation_split=0.33, batch_size=10, epochs=100)


# In[15]:


#listing all data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


print(model_history.history.keys())
#summerize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[17]:


#summerize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[18]:


#making prdictions
#Predicting the test set reults 
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)


# In[19]:


#making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, y_pred)


# 

# In[20]:


cm


# In[21]:


#calculate the accuracy
from sklearn.metrics import accuracy_score
score= accuracy_score(y_pred, Y_test)


# In[22]:


score


# In[ ]:




