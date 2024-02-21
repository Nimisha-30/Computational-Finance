# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:02:09 2023

@author: 20pt20
"""

# x=[1, 2, 3, 4, 5]
# y=[3, 4, 2, 4, 5]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("C://PSG//Sem8//20XTO1 Computational Finance//Lab//1//CSVForMonth.csv", index_col=False)

def linear_regression(df1, df2):
    x=np.array(df1)
    y=np.array(df2)
    mean_x=sum(x)/len(x)
    mean_y=sum(y)/len(y)
    x_mean_x=[]
    for i in range(len(x)):
        x_mean_x.append(x[i]-mean_x)
    y_mean_y=[]
    for i in range(len(y)):
        y_mean_y.append(y[i]-mean_y)
    x_mean_x_2=[]
    for i in x_mean_x:
        x_mean_x_2.append(i**2)
    prod=[]
    for i in range(len(x_mean_x)):
        prod.append(x_mean_x[i]*y_mean_y[i])
    m=sum(prod)/sum(x_mean_x_2)
    #print(m)
    #c=y[0]-x[0]*m
    c=mean_y-mean_x*m
    #print(c)
    return m, c

def predict_values(x_test, m, c):
    y_pred=[]
    for i in range(len(x_test)):
        y=m*x_test[i]+c
        y_pred.append(round(y))
    return y_pred

# def construct_predict(x, y, x_test):
#     mean_x=sum(x)/len(x)
#     mean_y=sum(y)/len(y)
#     x_mean_x=[]
#     for i in range(len(x)):
#         x_mean_x.append(x[i]-mean_x)
#     y_mean_y=[]
#     for i in range(len(y)):
#         y_mean_y.append(y[i]-mean_y)
#     x_mean_x_2=[]
#     for i in x_mean_x:
#         x_mean_x_2.append(i**2)
#     prod=[]
#     for i in range(len(x_mean_x)):
#         prod.append(x_mean_x[i]*y_mean_y[i])
#     m=sum(prod)/sum(x_mean_x_2)
#     c=mean_y-mean_x*m
#     y_pred=[]
#     for i in range(len(x_test)):
#         y=m*x_test[i]+c
#         y_pred.append(round(y))
#     return y_pred

def construct_predict(df1, df2):
    x=np.array(df1)
    y=np.array(df2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
    mean_x=sum(x_train)/len(x_train)
    mean_y=sum(y_train)/len(y_train)
    x_mean_x=[]
    for i in range(len(x_train)):
        temp=x_train[i]-mean_x
        x_mean_x.append(temp)
    y_mean_y=[]
    for i in range(len(y_train)):
        temp=y_train[i]-mean_y
        y_mean_y.append(temp)
    x_mean_x_2=[]
    for i in x_mean_x:
        temp=i**2
        x_mean_x_2.append(temp)
    prod=[]
    for i in range(len(x_mean_x)):
        temp=x_mean_x[i]*y_mean_y[i]
        prod.append(temp)
    m=sum(prod)/sum(x_mean_x_2)
    c=mean_y-mean_x*m
    print(m, c)
    y_pred=[]
    for i in range(len(x_test)):
        y=m*x_test[i]+c
        y_pred.append(round(y, 2))
    return y_pred, y_test

def t_test(y_pred, y_test, alpha=0.1):
    n=len(y_pred)
    error=[]
    for i in range(len(y_pred)):
        error.append(y_pred[i]-y_test[i])
    error_square=[]
    for j in error:
        error_square.append(j**2)
    sum_error_square=sum(error_square)
    sd=(sum_error_square/(n-2))**(1/2)
    t_stat=sum(error)/sd
    print(t_stat)
    critical_value=1.886
    if abs(t_stat) > critical_value:
        print(f"Reject the null hypothesis at {alpha} significance level.")
        print("There is evidence of a significant relationship between the variables.")
    else:
        print(f"Fail to reject the null hypothesis at {alpha} significance level.")
        print("There may not be a significant relationship between the variables.")
    return t_stat

# m, c=linear_regression(x, y)
# print(m, c)
# x_test=[6, 7, 8]
# y_pred=predict_values(x_test, m, c)
# print(y_pred)
# print()
# print(construct_predict(x, y, x_test))

# print(linear_regression(df['Open'], df['High']))
y_pred, y_test=construct_predict(df['Open'], df['High'])
t=t_test(y_pred, y_test)