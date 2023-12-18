# Exponantial smoothing 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# We used the Holt-Winters Exponential Smoothing method for time series forecasting

#load th datasets
data = pd.read_csv("F:\Learning_Work\Vs_Work\DM_Project\AirPassengers.csv", parse_dates=True, index_col="Month")

# ploting the datasets into graph
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("Air Passenger Data")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()

# to decompose the time series into the trends, seasonal and components 
decomposition = seasonal_decompose(data, model="multiplicative")
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
# visualizing the patterns in the data
plt.subplot(411)
plt.plot(data, label="Original")
plt.legend(loc="best")
plt.subplot(412)
plt.plot(trend, label="Trend")
plt.legend(loc="best")
plt.subplot(413)
plt.plot(seasonal, label="Seasonal")
plt.legend(loc="best")
plt.subplot(414)
plt.plot(residual, label="Residual")
plt.legend(loc="best")
plt.tight_layout()


# defining the instance of exponantial smoothing and specifing the trends and seasonal components and periods 
model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=12)
model_fit = model.fit()

# Make forecasts  upto 12 time points
forecast = model_fit.forecast(steps=12)


# visualize
plt.figure(figsize=(12, 6))
plt.plot(data, label="Original")
plt.plot(forecast, label="Forecast")
plt.title("Air Passenger Data and Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()

