# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# df = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\CSVForMonthJan16.csv', parse_dates = ['Month'], index_col = ['Month'])
df = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\CSVForMonthJan16.csv')
closeDF = df[['Month', 'Close']].copy()
closeDF['Month'] = pd.to_datetime(closeDF['Month'], format = '%b-%d')

closeDF['Close'].plot()
plt.title("Close Prices")
plt.show()

X = closeDF['Close'].values
size = int(len(X) * 0.66)
train, test = X[0: size], X[size: len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order = (20, 1, 0))
    modelFit = model.fit()
    output = modelFit.forecast()
    yHat = output[0]
    predictions.append(yHat)
    obs = test[t]
    history.append(obs)
    print('Predicted = %f, Expected = %f' % (yHat, obs))
    
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.2f' % rmse)

temp = [None] * (size - 1)
temp.extend(predictions)

plt.plot(X, label = 'Actual')
plt.plot(temp, color = 'red', label = 'Predicted')
plt.title('Predictions')
plt.show()

model = ARIMA(history, order = (20, 1, 0))
modelFit = model.fit()
forecast = modelFit.predict(start = len(history), end = len(history) + 10, typ = 'levels')

temp = [None] * (len(X) - 1)
temp.extend(forecast[1:])
plt.plot(X, label = 'Actual')
plt.plot(temp, color = 'red', label = 'Forecast')
plt.title('Forecast')
plt.show()

for t in range(20):
    model = ARIMA(history, order = (20, 1, 0))
    modelFit = model.fit()
    output = modelFit.forecast()
    yHat = output[0]
    predictions.append(yHat)
    history.append(yHat)

temp = [None] * (size - 1)
temp.extend(predictions)
plt.plot(X, label = 'Actual')
plt.plot(temp, color = 'red', label = 'Forecast')
plt.title('Forecast 2')
plt.show()