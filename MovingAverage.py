import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\CSVForMovingAverage.csv', index_col = None)

movingAvg = [None, None, None]
for i in range(3, len(df['Close'])):
    total = 0
    total += df['Close'][i] + df['Close'][i - 1] + df['Close'][i - 2] + df['Close'][i - 3]
    movingAvg.append(total / 4)

centeredMovingAvg = []

for i in range(2, len(movingAvg)):
    if movingAvg[i] != None and movingAvg[i - 1] != None:
        centeredMovingAvg.append((movingAvg[i] + movingAvg[i - 1]) / 2)
    else:
        centeredMovingAvg.append(None)
centeredMovingAvg.append(None)
centeredMovingAvg.append(None)

plt.plot(df['Month'], df['Close'], label = 'Close')
plt.plot(df['Month'], movingAvg, label = 'Moving Average')
plt.plot(df['Month'], centeredMovingAvg, label = 'Centered Moving Average')
plt.legend()
plt.xlabel('Month-Year')
plt.ylabel('Close Price')