# ANN (Artificial Neural Network)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the Air Passenger dataset
data = pd.read_csv("F:\Learning_Work\Vs_Work\DM_Project\AirPassengers.csv")
passengers = data["Passengers"].values.astype(float)

# Normalize data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))        #Min-Max scaling to normalize the passenger data to the range [0, 1].
passengers = scaler.fit_transform(passengers.reshape(-1, 1))

# Split the dataset into training and testing sets
train_size = int(len(passengers) * 0.67)         # split the dataset for training and testing 67% for training and rest for testing
test_size = len(passengers) - train_size
train, test = passengers[0:train_size, :], passengers[train_size:len(passengers), :]

# Function to create time series data
def create_dataset(dataset, look_back=1):  #to create time series data with a specified look-back period.
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Create and compile the ANN model
# Create a simple Artificial Neural Network (ANN) model using Keras
#  with one input layer, one hidden layer with 8 neurons (ReLU activation), and one output layer.
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# Define early stopping to prevent overfitting
#Define early stopping to monitor the validation loss and stop training if there's no improvement after 10 steps.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2, validation_data=(testX, testY), callbacks=[early_stopping])

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions to the original scale
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error (RMSE)
trainScore = sqrt(mean_squared_error(trainY[0], trainPredict[:, 0])) 
print(f"Train Score: {trainScore:.2f} RMSE")
testScore = sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print(f"Test Score: {testScore:.2f} RMSE")

# Plot the results
trainPredictPlot = np.empty_like(passengers)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(passengers)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(passengers) - 1, :] = testPredict

plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(passengers), label="Actual Passengers")
plt.plot(trainPredictPlot, label="Training Predictions")
plt.plot(testPredictPlot, label="Testing Predictions")
plt.legend()
plt.show()
