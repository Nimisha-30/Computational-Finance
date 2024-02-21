import pandas as pd
import matplotlib.pyplot as plt

def residualSumOfSquares(x, y, pred):
    residual = []
    rss = 0
    for i in range(len(x)):
        residual.append(y[i] - pred[i])
        rss += residual[i] ** 2
    return rss

def tTest(sxx, syy, n, b1):
    # DO THE T-TEST FOR THE ACTUAL MEAN AND PREDICTED MEAN
    # https://vitalflux.com/linear-regression-t-test-formula-example/
    se = ((syy / (n - 2)) ** 0.5) / (sxx ** 0.5)
    t = b1 / se
    return t

def regression(df, x, y):
    print("Regression for", x, "-", y)
    # Estimate coefficients
    xMean = df[x].mean()
    yMean = df[y].mean()
    
    sxy = 0
    sxx = 0
    syy = 0
    
    for i in range(len(df[x])):
        sxy += (df[x][i] - xMean) * (df[y][i] - yMean)
        sxx += (df[x][i] - xMean) * (df[x][i] - xMean)
        syy += (df[y][i] - yMean) * (df[y][i] - yMean)
    
    b1 = sxy / sxx
    b0 = yMean - xMean * b1
    print("B0:", b0, "\tB1:", b1)
    
    yPred = []
    for i in range(len(df[x])):
        yPred.append(b0 + b1 * df[x][i])
    
    plt.scatter(df[x], df[y])
    plt.plot(df[x], yPred, color = 'r')
    
    # RSS
    print("RSS =", residualSumOfSquares(df[x], df[y], yPred))
    print("T-Value =", tTest(sxx, syy, len(df[x]), b1))
    

# MAIN
df = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\CSVForMonthJan16.csv')

regression(df, 'Open', 'Close')
plt.show()