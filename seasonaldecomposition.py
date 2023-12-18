# seasonal decomposition 

# it is used to break down a time series into different components, typically trend, seasonality, and remainder (residual). 
# The purpose of seasonal decomposition is to understand the underlying patterns and structures within the time series data, 
# which can help in making more accurate predictions or in gaining insights into the data.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the Air Passenger dataset
data = pd.read_csv("F:\Learning_Work\Vs_Work\DM_Project\AirPassengers.csv", parse_dates=True, index_col="Month")
passengers = data["Passengers"]

# Perform STL decomposition         
stl = STL(passengers, seasonal=13)  # You can experiment with different seasonal values
result = stl.fit()

# Extract the trend and seasonal components
trend = result.trend
seasonal = result.seasonal

# Forecast future values using the trend component
future_periods = 12  # Forecast 12 months into the future
forecast = trend[-1] + seasonal[-future_periods:]

# Calculate root mean squared error  to evaluate the performance of the model
test = passengers[-future_periods:]
rmse = sqrt(mean_squared_error(test, forecast))
print(f"Test RMSE: {rmse:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(passengers, label="Actual Data")
plt.plot(forecast, label="Forecast")
plt.legend()
plt.title("Air Passenger Forecast with STL Decomposition")
plt.show()
