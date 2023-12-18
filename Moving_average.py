# Simple moving average Algorithm (SMA) 
# based on observations k is the size of window to determine the average
# A simple moving average tells us the unweighted mean of the previous K data points.
# The more the value of K the more smooth is the curve, but increasing K decreases accuracy.
# If the data points are p1,  p2, . . . , pn then we calculate the simple moving average.

# we can calculate the moving average using .rolling() method. 
# This method provides rolling windows over the data, and we can use the mean function over these windows to calculate moving averages.
# The size of the window is passed as a parameter in the function .rolling(window).

# importing Libraries
# importing pandas as pd
import pandas as pd

# importing numpy as np
# for Mathematical calculations
import numpy as np

# importing pyplot from matplotlib as plt
# for plotting graphs
import matplotlib.pyplot as plt
plt.style.use('default')

# importing time-series data
reliance = pd.read_csv('F:\Learning_Work\Vs_Work\DM_Project\RELIANCE.NS_.csv', index_col='Date',
					parse_dates=True)

# Printing dataFrame
print('Head of the Dataframe.')
print('||||--------------------------------------------------------------------------------------||||\n')
print(reliance.head())


# updating our dataFrame to have only
# one column 'Close' as rest all columns are of no use for us at the moment using .to_frame() to convert pandas series into dataframe.
reliance = reliance['Close'].to_frame()
print('Close column of the Dataframe.')
print('||||--------------------------------------------------------------------------------------||||\n')
print(reliance)

# calculating simple moving average
# using .rolling(window).mean() ,
# with window size = 30
reliance['SMA30'] = reliance['Close'].rolling(30).mean()

# removing all the NULL values using 
# dropna() method
reliance.dropna(inplace=True)

# printing Dataframe
print('Dataframe after null values removed and SMA column added:')
print('||||--------------------------------------------------------------------------------------||||\n')
print(reliance)

# plotting Close price and simple
# moving average of 30 days using .plot() method
reliance[['Close', 'SMA30']].plot(label='RELIANCE', 
								figsize=(16, 8))

plt.show()
