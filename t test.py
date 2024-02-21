# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:39:01 2024

@author: 20pt20
"""
from scipy.stats import ttest_1samp
def t_test(residuals):
    print("Null Hypothesis - H0:")
    print("The values of", " do not affect the values of")
    print("Alternative Hypothesis - H1:")
    print("The values of", " affect the values of")
    t, p= ttest_1samp(a=residuals, popmean=0)
    print("t:", t)
    print("p:", p)
    alpha=0.05
    if p<=alpha:
        print("Reject Null Hypothesis (H0) as p < alpha. The model is significant.")
    else:
        print("Accept Null Hypothesis (H0). The model may not be significant.")


def t_test1(data, mu):
    mean_data=sum(data)/len(data)
    n=len(data)
    residuals=[]
    for i in range(len(data)):
        residuals.append(data[i]-mean_data)
    residuals_square=[]
    for j in residuals:
        residuals_square.append(j**2)
    sum_residuals_square=sum(residuals_square)
    sd_square=sum_residuals_square/(n-1)
    sd=sd_square**(1/2)
    #print(sd)
    num=(mean_data-mu)
    denom=sd/(n**(1/2))
    t=num/denom
    print(t)
    alpha=0.05
    if p<=alpha:
        print("Reject Null Hypothesis (H0) as p < alpha. The model is significant.")
    else:
        print("Accept Null Hypothesis (H0). The model may not be significant.")

data=[1, 2, 3, 4, 5]
t_test(data)
t_test1(data, 0)