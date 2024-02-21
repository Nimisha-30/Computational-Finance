# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:51:33 2023

@author: 20pt20
"""

x=[1, 2, 3, 4, 5]
y=[2, 4, 6, 8, 10]

def corr(x, y):
    n=len(x)
    sum_x=sum(x)
    sum_y=sum(y)
    x_y=[]
    for i in range(len(x)):
        x_y.append(x[i]*y[i])
    sum_x_y=sum(x_y)
    x_2=[]
    for i in range(len(x)):
        x_2.append(x[i]**2)
    sum_x_2=sum(x_2)
    y_2=[]
    for i in range(len(y)):
        y_2.append(y[i]**2)
    sum_y_2=sum(y_2)
    dividend=(n*sum_x_y)-(sum_x*sum_y)
    d1=(n*sum_x_2)-(sum_x**2)
    d2=(n*sum_y_2)-(sum_y**2)
    divisor=d1*d2
    divisor=divisor**(1/2)
    corr=dividend/divisor
    return round(corr, 2)

print(corr(x, y))