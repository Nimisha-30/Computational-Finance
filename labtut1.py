# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:55:45 2023

@author: 20pt20
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp

df=pd.read_csv("C://PSG//Sem8//20XTO1 Computational Finance//Lab//1//CSVForMonth.csv", index_col=False)
# print(df.head(5))

# https://byjus.com/maths/correlation/
def corr(df1, df2):
    n=len(df1)
    sum_df1=sum(df1)
    sum_df2=sum(df2)
    df['df1*df2']=df1*df2
    sum_df1df2=sum(df['df1*df2'])
    df['df1square']=df1.pow(2)
    df['df2square']=df2.pow(2)
    sum_df1square=sum(df['df1square'])
    sum_df2square=sum(df['df2square'])
    dividend=(n*sum_df1df2)-(sum_df1*sum_df2)
    d1=(n*sum_df1square)-(sum_df1*sum_df1)
    d2=(n*sum_df2square)-(sum_df2*sum_df2)
    divisor=d1*d2
    divisor=divisor**(1/2)
    corr=dividend/divisor
    return round(corr, 2)

# https://realpython.com/linear-regression-in-python/
def linear_regression(df1, df2):
    x=np.array(df1)
    y=np.array(df2)
    if len(x.shape)==1:
        x=x.reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
    model=LinearRegression().fit(x_train, y_train)
    y_pred=model.predict(x_test)
    r_sq=model.score(x_train, y_train)
    print("R square: ", round(r_sq, 2))
    # Plotting true values vs predicted values
    if x.shape[1]==1:
        plt.scatter(x_test, y_test, color='black')
        plt.plot(x_test, y_pred,color='blue')
        plt.show()
    return y_test, y_pred

def t_test(y_test, y_pred):
    print("Null Hypothesis - H0:")
    print("The values of", " do not affect the values of")
    print("Alternative Hypothesis - H1:")
    print("The values of", " affect the values of")
    residuals=y_test-y_pred
    t, p= ttest_1samp(a=residuals, popmean=0)
    print("t:", t)
    print("p:", p)
    alpha=0.05
    if p<=alpha:
        print("Reject Null Hypothesis (H0) as p < alpha. The model is significant.")
    else:
        print("Accept Null Hypothesis (H0). The model may not be significant.")

# LINEAR REGRESSION ANALYSIS

openhigh=corr(df['Open'], df['High'])
print("Correlation Coefficient for Open-High:", openhigh)
y_test, y_pred=linear_regression(df['Open'], df['High'])
t_test(y_test, y_pred)
print()

# openlow=corr(df['Open'], df['Low'])
# print("Correlation Coefficient for Open-Low:", openlow)
# linear_regression(df['Open'], df['Low'])
# print()

# openclose=corr(df['Open'], df['Close'])
# print("Correlation Coefficient for Open-Close:", openclose)
# linear_regression(df['Open'], df['Close'])
# print()

# highlow=corr(df['High'], df['Low'])
# print("Correlation Coefficient for High-Low:", highlow)
# linear_regression(df['High'], df['Low'])
# print()

# highclose=corr(df['High'], df['Close'])
# print("Correlation Coefficient for High-Close:", highclose)
# linear_regression(df['High'], df['Close'])
# print()

# lowclose=corr(df['Low'], df['Close'])
# print("Correlation Coefficient for Low-Close:", lowclose)
# linear_regression(df['Low'], df['Close'])
# print()