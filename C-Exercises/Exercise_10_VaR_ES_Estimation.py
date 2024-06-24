#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:35:22 2022

QF 13: Rounak Bastola, Adrian Beer
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def VaR_ES_historic (x_data, l, alpha):
    x_loss = l(x_data)
    x_sort = np.sort(x_loss)
    VaR = x_sort[int(np.ceil(alpha*len(x_sort)))]
    ES = np.mean(x_sort[int(np.ceil(len(x_sort)*(alpha))):])
    return [VaR, ES]

def VaR_log_normal(s,alpha):
    x = np.diff(np.log(s))
    mu = x.mean()
    std = x.std()
    VaR = s[-1]*(1 - np.exp(mu + std*norm.ppf(1-alpha)))
    return VaR


if __name__ == "__main__":
    data = np.flip(np.genfromtxt('C-Exercises/dax_data.csv',delimiter=';',usecols=4,skip_header=1))

    alpha = [0.9, 0.95]
    n=247
    l = lambda k: -np.diff(k)
    VaR_ES, VaR_log = [],[]
    for a in alpha:
        VaR_ES.append(np.array([VaR_ES_historic(data[m:m+n], l, a) for m in range(0,len(data)-n)]))
        VaR_log.append(np.array([VaR_log_normal(data[m:m+n], a) for m in range(0,len(data)-n)]))
    
    loss = -np.diff(data)[n-1:]
    for i in range(len(alpha)):
        print('We expected %s Violations for %s%%.'%(int((1-alpha[i])*len(loss)),int(alpha[i]*100)))
        print('For Historic, there were %s Vioations.'%(len(VaR_ES[i].T[0][np.where(loss>VaR_ES[i].T[0])])))
        print('For Log Normal, there were %s Vioations.'%(len(VaR_log[i].T[np.where(loss>VaR_log[i].T)])))
    fig, axis= plt.subplots(2,1)
    for i in range(len(alpha)):
        axis[i].set_title('VaR, ES for %s%%'%int(alpha[i]*100))
        axis[i].plot(VaR_ES[i].T[0],label='Historic VaR')
        axis[i].plot(VaR_ES[i].T[1], label='Historic ES')
        axis[i].plot(VaR_log[i].T,label='logNormal VaR')
    axis[0].legend()
    plt.tight_layout()