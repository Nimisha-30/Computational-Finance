# https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

def checkStationarity(df):
    rollingMean = df.rolling(window = 10).mean()
    rollingSTD = df.rolling(window = 10).std()
    
    # Plot
    plt.plot(df, color = 'blue', label = 'Original')
    plt.plot(rollingMean, color = 'red', label = 'Rolling Mean')
    plt.plot(rollingSTD, color = 'black', label = 'Rolling Standard Deviation')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean and Rolling Standard Deviation')
    plt.show(block = False)
    
    # Dickey-Fuller Test
    result = adfuller(df['Close'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-Value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

df = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\CSVForMonthJan16.csv', parse_dates = ['Month'], index_col = ['Month'])
closeDF = df[['Close']].copy()

# Before building a model, we must ensure that the time series is stationary
checkStationarity(closeDF)

# The time series is not stationary
# i) Subtracting Rolling Mean
closeDFLog = np.log(closeDF)
rollingMean = closeDFLog.rolling(window = 10).mean()
closeDFLog_minusMean = closeDFLog - rollingMean
closeDFLog_minusMean.dropna(inplace = True)
checkStationarity(closeDFLog_minusMean)
# The time series is now stationary

# ii) Exponential Decay
rollingMeanExpDecay = closeDFLog.ewm(halflife = 10, min_periods = 0, adjust = True).mean()
dfLogExpDecay = closeDFLog - rollingMeanExpDecay
dfLogExpDecay.dropna(inplace = True)
checkStationarity(dfLogExpDecay)
# Subtracting Rolling Mean is more stationary

decomposition = seasonal_decompose(closeDFLog, period = 1)
model = ARIMA(closeDF, order = (2, 1, 2))
results = model.fit()
plt.plot(closeDFLog_minusMean)
plt.plot(results.fittedvalues, color = 'red')
plt.show()

predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy = True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(closeDFLog['Close'].iloc[0], index = closeDFLog.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value = 0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(closeDF)
plt.plot(predictions_ARIMA)
plt.show()