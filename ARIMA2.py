# https://blog.quantinsti.com/forecasting-stock-returns-using-arima-model/#:~:text=Stock%20market%20forecasting%20has%20always,Moving%20Average%20(ARIMA)%20model.
# https://analyticsindiamag.com/quick-way-to-find-p-d-and-q-values-for-arima/#:~:text=Draw%20a%20partial%20autocorrelation%20graph,to%20the%20ACF%20is%20q.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

def checkStationarity(timeSeries):
    rolMean = timeSeries.rolling(12).mean()
    rolSTD = timeSeries.rolling(12).std()
    
    adft = adfuller(timeSeries, autolag = 'AIC')
    output = pd.Series(adft[0:4], index = ['Test Statistics', 'p-Value', 'No. of lags used', 'No. of observations used'])
    for key, values in adft[4].items():
        output['Critical Value (%s)'%key] = values
    print(output)

dateparse = lambda dates: pd.datetime.strptime(dates, '%b-%y')
stockData = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\CSVForMonthJan16.csv', index_col = 'Month', parse_dates = ['Month'], date_parser = dateparse).fillna(0)
dfClose = stockData[['Close']]

plt.grid(True)
plt.xlabel('Month')
plt.ylabel('Close Prices')
plt.plot(stockData['Close'])
plt.title('Closing Price Graph')
plt.show()

checkStationarity(dfClose)

result = seasonal_decompose(dfClose, model = 'multiplicative', period = 12)

dfLog = np.log(dfClose)
movingAvg = dfLog.rolling(12).mean()
stdDev = dfLog.rolling(12).std()

trainData, testData = dfLog[3:int(len(dfLog) * 0.8)], dfLog[int(len(dfLog) * 0.8):]

plt.grid(True)
plt.xlabel('Months')
plt.ylabel('Closing Prices')
plt.plot(dfLog, 'green', label = 'Train Data')
plt.plot(testData, 'blue', label = 'Test Data')
plt.legend()
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(dfLog)
ax1.set_title('Log Series')
ax2.plot(dfLog.diff())
ax2.set_title('1st Order Differencing')
ax3.plot(dfLog.diff().diff())
ax3.set_title('2nd Order Differencing')
plt.show()
# d = 1

plot_pacf(dfLog.diff().dropna())
plt.show()
# p = 1

plot_acf(dfLog.diff().dropna())
plt.show()
# q = 1

testData = testData.reset_index()
print(testData['Month'])

model = ARIMA(trainData, order = (1, 1, 1))
fitted = model.fit()
print(fitted.summary())
plt.plot(fitted.predict(start = len(trainData), end = len(trainData) + 10))
plt.title('Prediction')
plt.show()