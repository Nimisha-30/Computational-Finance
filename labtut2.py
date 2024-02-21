# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:22:14 2024

@author: 20pt20
"""

import pandas as pd

df=pd.read_csv("C://PSG//Sem8//20XTO1 Computational Finance//Lab//2//CSVForMonth.csv", index_col=False)
#print(df.head(5))

def four_quarter_moving_total(df):
    quarter_avg=[]
    i=0
    while i<=(len(df)-2):
        avg=(df[i]+df[i+1]+df[i+2])/3
        quarter_avg.append(avg)
        i+=3
    # print(quarter_avg, "\n")
    moving_total=[]
    for i in range(len(quarter_avg)-3):
        moving_sum=quarter_avg[i]+quarter_avg[i+1]+quarter_avg[i+2]+quarter_avg[i+3]
        moving_total.append(moving_sum)
    # print(moving_total, "\n")
    return quarter_avg, moving_total

def four_quarter_moving_average(moving_total):
    moving_avg=[]
    for i in range(len(moving_total)):
        moving_avg.append(moving_total[i]/4)
    # print(moving_avg, "\n")
    return moving_avg

def center_moving_average(moving_avg):
    centered_mvavg=[]
    for i in range(len(moving_avg)-1):
        center=(moving_avg[i]+moving_avg[i+1])/2
        centered_mvavg.append(center)
    # print(centered_mvavg, "\n")
    return centered_mvavg

def calculate_percentage(quarter_avg, centered_mvavg):
    percentage=[]
    n=len(quarter_avg)
    quarter_avg.remove(quarter_avg[n-1])
    quarter_avg.remove(quarter_avg[n-2])
    quarter_avg.remove(quarter_avg[1])
    quarter_avg.remove(quarter_avg[0])
    for i in range(len(centered_mvavg)):
        percent=quarter_avg[i]/centered_mvavg[i]*100
        percentage.append(percent)
    # print(percentage, "\n")
    return percentage

def modified_mean(percentage):
    col1=[]
    col2=[]
    col3=[]
    col4=[]
    i=0
    while i<=len(percentage)-1:
        col3.append(percentage[i])
        i=i+4
    i=1
    while i<=len(percentage)-1:
        col4.append(percentage[i])
        i+=4
    i=2
    while i<=len(percentage)-1:
        col1.append(percentage[i])
        i+=4
    i=3
    while i<=len(percentage)-1:
        col2.append(percentage[i])
        i+=4
    # print(col1, col2, col3, col4)
    col1.remove(max(col1))
    col1.remove(min(col1))
    col2.remove(max(col2))
    col2.remove(min(col2))
    col3.remove(max(col3))
    col3.remove(min(col3))
    col4.remove(max(col4))
    col4.remove(min(col4))
    # print(col1, col2, col3, col4)
    mm1=sum(col1)/2
    mm2=sum(col2)/2
    mm3=sum(col3)/2
    mm4=sum(col4)/2
    mm_total=mm1+mm2+mm3+mm4
    adjusting_constant=400/mm_total
    seasonal_index1=mm1*adjusting_constant
    seasonal_index2=mm2*adjusting_constant
    seasonal_index3=mm3*adjusting_constant
    seasonal_index4=mm4*adjusting_constant
    total_seasonal_index=seasonal_index1+seasonal_index2+seasonal_index3+seasonal_index4
    mean_seasonal_indices=total_seasonal_index/4
    # print(seasonal_index1, "\t", seasonal_index2, "\t", seasonal_index3, "\t", seasonal_index4)
    # print(mean_seasonal_indices)
    return [seasonal_index1, seasonal_index2, seasonal_index3, seasonal_index4]

qa, mt=four_quarter_moving_total(df['Close'])
mt2=four_quarter_moving_average(mt)
mt3=center_moving_average(mt2)
p=calculate_percentage(qa, mt3)
lst=modified_mean(p)
print(lst)