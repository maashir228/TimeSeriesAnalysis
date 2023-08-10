#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[2]:


Metadf = pd.read_csv('symbols_valid_meta.csv')
Metadf


# In[10]:


aau = pd.read_csv('stocks/aau.csv')
aau


# In[24]:


aau["Date"] = pd.to_datetime(aau["Date"])
aau['Day'] = aau['Date'].dt.day
aau


# In[46]:


aau['Month'] = aau['Date'].dt.month
aau


# In[47]:


train_dates = pd.to_datetime(aau['Date'])
print(train_dates.tail(15))


# In[51]:


cols = list(aau)[1:6]
cols2 = list(aau)[7:9]
#Date and volume columns are not used in training. 
print(cols)
print(cols2)
cols_for_training = cols+cols2


# In[52]:


df_for_training = aau[cols_for_training].astype(float)
df_for_training


# In[53]:


scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


# In[54]:


trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 30  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))


# In[55]:


#In my case, trainX has a shape (12809, 14, 5). 
#12809 because we are looking back 14 days (12823 - 14 = 12809). 
#Remember that we cannot look back 14 days until we get to the 15th day. 
#Also, trainY has a shape (12809, 1). Our model only predicts a single value, but 
#it needs multiple variables (5 in my example) to make this prediction. 
#This is why we can only predict a single day after our training, the day after where our data ends.
#To predict more days in future, we need all the 5 variables which we do not have. 
#We need to predict all variables if we want to do that. 

# define the Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
history = model.fit(trainX, trainY, epochs=20, batch_size=16, validation_split=0.3, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

#Predicting...
#Libraries that will help us extract only business days in the US.
#Otherwise our dates would be wrong when we look back (or forward).  
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
#Remember that we can only predict one day in future as our model needs 5 variables
#as inputs for prediction. We only have all 5 variables until the last day in our dataset.
n_past = 16
n_days_for_prediction=15  #let us predict past 15 days

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()
print(predict_period_dates)

#Make prediction
prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]


# In[73]:


# Convert timestamp to month
forecast_month = []
for time_i in predict_period_dates:
    forecast_month.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_month), 'Open':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])
df_forecast


# In[76]:


# Filter data from January 2020 onwards
original = aau[['Date','Month', 'Open']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2020-01-01']

# Create a line plot
plt.figure(figsize=(10, 6))

# Plot the original stock prices
sns.lineplot(x='Date', y='Open', data=original, label='Original Data')

# Plot the forecasted stock prices
sns.lineplot(x='Date', y='Open', data=df_forecast, label='Forecasted Data')

plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (Open)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

