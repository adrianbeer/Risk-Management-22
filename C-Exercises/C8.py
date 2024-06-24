#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 08:49:23 2022

@author: me
"""
import numpy as np
from scipy.stats import norm
import pandas  as pd
import matplotlib.pyplot as plt


def VaR_ES_var_covar (x_data, c, w, alpha):
    mu = x_data.mean(1)
    cov = np.cov(x_data)
    VaR = -(c + np.dot(w,mu)) + np.sqrt(np.dot(np.dot(w,cov),w))*norm.ppf(alpha)
    ES = -(c + np.dot(w,mu)) + (np.sqrt(np.dot(np.dot(w,cov),w))*norm.pdf(norm.ppf(alpha))/(1-alpha))
    return np.array([VaR, ES])




stocks = ['wkn_519000_historic.csv','wkn_716460_historic.csv','wkn_543900_historic.csv','wkn_766403_historic.csv','wkn_723610_historic.csv']
#bmw,sap,continental,volkswagen, siemens
s = np.array(pd.concat((pd.read_csv(s, delimiter=';', usecols=[4]).replace(',','.',regex=True).astype(float) for s in stocks),axis=1, ignore_index=True)).T
s = np.flip(s,axis=1)
x = np.diff(np.log(s))
alpha_p = [38, 31, 24, 50, 22]
w = (alpha_p*s.T).T
alpha = 0.98
c = 0
m = 254
n = 252
VaR_ES = np.array([VaR_ES_var_covar(x[:,m+i:m+n+1+i], c, w[:,m+i], alpha) for i in range(len(x[0,m:-n]))])#np.empty(len(x[:,m:]))  
  

plt.plot(VaR_ES.T[0],label='VaR')
plt.plot(VaR_ES.T[1], label='ES')
plt.legend()
plt.title('Value at Risk and Expected Shortfall')
    