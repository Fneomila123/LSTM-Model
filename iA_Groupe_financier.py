#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libraries 


# In[132]:


import math
import numpy as np 
import pandas as pd 
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
plt.style.use("fivethirtyeight")


# In[18]:


#get the stock 


# In[126]:


df= pdr.DataReader('IAG.TO', data_source='yahoo', start="2010-02-01", end='2022-02-18')


# In[122]:


df


# In[84]:


df.shape


# In[131]:


df.info()
#df.shape


# In[133]:


#visualize the closing price 
plt.figure(figsize=(16, 9))
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=19)
plt.ylabel('Close Price CAD', fontsize= 19)


# In[35]:


#In the y-axis is in milliion CAD 


# In[36]:


#Now, let's focus on the Close values, to be able to do it, we have  to create a new daa frame w.r.t the Close.


# In[38]:


DataClose= df.filter(['Close'])
dataset= DataClose.values #convert the dataframe for Close into numpy array (like a matrix (3025, 1))


# In[39]:


dataset


# In[105]:


#Data train.
l= len(dataset)
training_data_len =math.ceil( l*.8) #80% of the data to be trained
training_data_len


# In[44]:


#According to the data frame, it seems that the data as a large number, we have to scale it by MinMax


# In[47]:


scaler=MinMaxScaler(feature_range=(0,1))
scaled_data= scaler.fit_transform(dataset)
scaled_data


# In[57]:


#create the traning data set 
train_data= scaled_data[0: trainning_data_len, :]
#split the data into x_train andy_train 
x_train = []
y_train = []
#create a foor loop to see what happend in 60 days 

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60: i, 0])
    y_train.append(train_data[i, 0])
    if i<=60:
        print(x_train)
        print(y_train)


# In[59]:


#convert x_train and y_train intto numpy array 
x_train, y_train = np.array(x_train), np.array(y_train)


# In[61]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[62]:


x_train.shape


# In[76]:


#Build our LSTM Model 
model= Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


# In[79]:


#Compile the mode 
model.compile(optimizer='adam', loss='mean_squared_error')


# In[81]:


#train the model 
model.fit(x_train, y_train, batch_size= 1, epochs= 1)


# In[88]:


#create the test data set
#Create a new array containing scaled values for index 2360 to 3025
test_data = scaled_data[trainning_data_len -60: , :]
#Create the data sets x_test and y_test
x_test =[]
y_test= dataset[trainning_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# In[94]:


#Convert Data to an np array
x_test=np.array(x_test)
#reshape the data 
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[96]:


#Get th model predicted price values 
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)


# In[98]:


#RMSE calculating 
rmse= np.sqrt(np.mean(prediction - y_test)**2)
rmse


# In[109]:


#Plot the data 
train= DataClose[: training_data_len]
valid= DataClose[training_data_len:]
valid["Prediction"] = prediction
#visualize the data
plt.figure(figsize=(16, 8))
plt.title("Model")
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Close Price  CAD", fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Prediction']])
plt.legend(["Train", "Val", "Prediction"], loc="lower right")
plt.show()


# In[110]:


#Valid and Predicted price 
valid 


# In[134]:


#Get the quote
iA_groupe_Fin = pdr.DataReader("IAG.To", data_source='yahoo', start= '2010-02-01', end="2022-01-18")
iA_groupe_Fin


# In[114]:


#create a new data frame
new_df= iA_groupe_Fin.filter(['Close'])


# In[115]:


#Get the last 60 days closing price and convert the dataframe in array
last_60_days= new_df[-60: ].values


# In[118]:


#scale the data to be in 0 and 1
last_60_days_scaled  = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[137]:


iA_groupe_Fin2 = pdr.DataReader("IAG.To", data_source='yahoo', start= '2022-02-18', end="2022-02-18")
print(iA_groupe_Fin2['Close'])

