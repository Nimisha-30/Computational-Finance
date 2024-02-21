import pandas as pd
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math

def constraintWeights(weights):
    return np.sum(weights) - 1.0

def minObjectiveFunction(weights, covarianceMatrix):
    return np.dot(weights, np.dot(covarianceMatrix, weights))
def constraintExpectedReturn(weights, expectedReturns, targetReturn):
    return np.dot(weights, expectedReturns) - targetReturn

def maxObjectiveFunction(weights, expectedReturns):
    return -np.dot(weights, expectedReturns)
def constraintVariance(weights, covarianceMatrix, targetVariance):
    return np.dot(weights, np.dot(covarianceMatrix, weights)) - targetVariance

# Minimizing Risk
def minRisk(mu, covarianceMatrix):
    print("Minimizing Risk")
    initialWeights = np.ones(len(mu)) / len(mu)
    constraints = ({'type': 'eq', 'fun': constraintWeights})
    result = minimize(minObjectiveFunction, initialWeights, args = (covarianceMatrix,), method = 'SLSQP', constraints = constraints)
    optimalWeights = result.x
   
    print('Optimal Weights:', optimalWeights)
    print('Portfolio Variance:', result.fun)
    print('Portfolio Expected Return', np.dot(optimalWeights, mu))
    
    return [optimalWeights, result.fun, np.dot(optimalWeights, mu)]

# Maximizing Return
def maxReturn(mu, covarianceMatrix):
    print("Maximizing Return")
    initialWeights = np.ones(len(mu)) / len(mu)
    constraints = ({'type': 'eq', 'fun': constraintWeights})
    result = minimize(maxObjectiveFunction, initialWeights, args = (mu,), method = 'SLSQP', constraints = constraints)
    optimalWeights = result.x
    
    print('Optimal Weights:', optimalWeights)
    print('Portfolio Variance:', -result.fun)
    print('Portfolio Expected Return', np.dot(optimalWeights, np.dot(covarianceMatrix, optimalWeights)))

    return [optimalWeights, -result.fun, np.dot(optimalWeights, np.dot(covarianceMatrix, optimalWeights))]

# Minimizing Risk with Fixed Target Return
def minRiskTargetReturn(mu, covarianceMatrix):
    print("Minimizing Risk with Fixed Target Return")
    targetReturn = 0.01
    initialWeights = np.ones(len(mu)) / len(mu)
    constraints = ({'type': 'eq', 'fun': lambda w: constraintExpectedReturn(w, mu, targetReturn)},
                  {'type': 'eq', 'fun': constraintWeights})
    result = minimize(minObjectiveFunction, initialWeights, args = (covarianceMatrix,), method = 'SLSQP', constraints = constraints)
    optimalWeights = result.x
   
    print('Optimal Weights:', optimalWeights)
    print('Portfolio Variance:', result.fun)
    print('Portfolio Expected Return', np.dot(optimalWeights, mu))
    
    return [optimalWeights, result.fun, np.dot(optimalWeights, mu)]

# Maximizing Return with Fixed Target Variance
def maxReturnTargetVariance(mu, covarianceMatrix):
    print("Maximizing Return with Fixed Target Variance")
    targetVariance = 0.01
    initialWeights = np.ones(len(mu)) / len(mu)
    constraints = ({'type': 'eq', 'fun': constraintWeights},
                  {'type': 'eq', 'fun': lambda w: constraintVariance(w, covarianceMatrix, targetVariance)})
    result = minimize(maxObjectiveFunction, initialWeights, args = (mu,), method = 'SLSQP', constraints = constraints)
    optimalWeights = result.x
    
    print('Optimal Weights:', optimalWeights)
    print('Portfolio Variance:', -result.fun)
    print('Portfolio Expected Return', np.dot(optimalWeights, np.dot(covarianceMatrix, optimalWeights)))

    return [optimalWeights, -result.fun, np.dot(optimalWeights, np.dot(covarianceMatrix, optimalWeights))]

# MAIN
dateparse = lambda x: datetime.strptime(x, '%B-%Y')

df1 = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\Automobile\\500182.csv', usecols = ['Month', 'Open Price', 'Close Price'], parse_dates = ['Month'], date_parser = dateparse)
df2 = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\Automobile\\500477.csv', usecols = ['Month', 'Open Price', 'Close Price'], parse_dates = ['Month'], date_parser = dateparse)
df3 = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\Automobile\\500520.csv', usecols = ['Month', 'Open Price', 'Close Price'], parse_dates = ['Month'], date_parser = dateparse)
df4 = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\Automobile\\500570.csv', usecols = ['Month', 'Open Price', 'Close Price'], parse_dates = ['Month'], date_parser = dateparse)
df5 = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\Automobile\\505200.csv', usecols = ['Month', 'Open Price', 'Close Price'], parse_dates = ['Month'], date_parser = dateparse)
df6 = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\Automobile\\532343.csv', usecols = ['Month', 'Open Price', 'Close Price'], parse_dates = ['Month'], date_parser = dateparse)
df7 = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\Automobile\\532500.csv', usecols = ['Month', 'Open Price', 'Close Price'], parse_dates = ['Month'], date_parser = dateparse)
df8 = pd.read_csv('Z:\\20PT01\\SEM8\\CF Lab\\Datasets\\Automobile\\532977.csv', usecols = ['Month', 'Open Price', 'Close Price'], parse_dates = ['Month'], date_parser = dateparse)
df = [df1, df2, df3, df4, df5, df6, df7, df8]
for d in df:
    d['Close Price'].plot()
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Close Prices of Automobile Stocks')
plt.show()

for i in range(8):
    df[i]['Rate of Return'] = (df[i]['Close Price'] - df[i]['Open Price']) / df[i]['Open Price']

mu = np.array([df[i]['Rate of Return'].mean() for i in range(8)])
sigma = np.array([math.sqrt(df[i]['Rate of Return'].var()) for i in range(8)])

rho = list()
covarianceMatrix = list()
for i in range(8):
    row = list()
    covRow = list()
    for j in range(8):
        r = np.corrcoef(df[i]['Rate of Return'], df[j]['Rate of Return'])[0][1]
        cov = r * sigma[i] * sigma[j]
        row.append(r)
        covRow.append(cov)
    rho.append(row)
    covarianceMatrix.append(covRow)
rho = np.array(rho)
covarianceMatrix = np.array(covarianceMatrix)

# print("Expected Returns:", mu)
# print("Variances:", sigma)
# print("Correlation Coefficients:", rho)
# print("Covariance Matrix:", covarianceMatrix)

minRiskOptimal = minRisk(mu, covarianceMatrix)
print("=================================================================================")
maxReturnOptimal = maxReturn(mu, covarianceMatrix)

# Plotting the Risk-Return Graph
print()
print(math.sqrt(minRiskOptimal[1]), minRiskOptimal[2])
print(math.sqrt(maxReturnOptimal[1]), maxReturnOptimal[2])
plt.scatter(math.sqrt(minRiskOptimal[1]), minRiskOptimal[2], label = 'Min Risk')
# plt.scatter(math.sqrt(maxReturnOptimal[1]), maxReturnOptimal[2], label = 'Max Return')
plt.scatter(mu, sigma, label = 'Single Stocks')
plt.grid(True)
plt.xlabel('Risk')
plt.ylabel('Return')
plt.legend()
plt.title('Risk-Return Graph')
plt.show()