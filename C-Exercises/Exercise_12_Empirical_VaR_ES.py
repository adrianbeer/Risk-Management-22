#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 23:59:36 2022

Group 13: Rounak Bastola, Adrain Beer
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate as integrate

def empirical_cdf(n):
    return np.random.normal(0,1,n)
    

n = np.array([10, 100, 1000])
alpha = np.array([.9, .975]) 
x = np.linspace(-4, 4, 10000)
y = norm.cdf(x)
fig, axis = plt.subplots(len(n),1)
for i in range(len(n)):
    axis[i].hist(empirical_cdf(n[i]),density=True,cumulative=True)
    axis[i].set_title(f'For n={n[i]}')
    if i==0:
        axis[i].plot(x,y,color='red', label = 'Normal CDF')
    else:
        axis[i].plot(x,y,color='red')
fig.legend()
fig.tight_layout()


for a in alpha:
    print(f'For alpha = {a}')
    for k in n:
        cdf= np.sort(empirical_cdf(k))
        VaR = cdf[int(a*len(cdf))]
        ES = np.mean(cdf[np.where(cdf>VaR)])
        print(f'For n={k}, the VaR is {VaR:.3f} and the ES is {ES:.3f}')
    VaRt = norm.ppf(a)
    ESt = integrate.quad(lambda b: norm.ppf(b),a,1)[0]*(1/(1-a))
    print(f'The theoretical values for VaR is {VaRt:.3f} and ES is {ESt:.3f}')
    

    
    