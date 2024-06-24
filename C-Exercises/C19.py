#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:38:06 2022

@author: me
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42069)

def Hill_Estimator(x, k):
    return k/np.sum((np.log(x[:k-1]) - np.log(x[k-1])))


def Hill_Plot(x, k):
    x = np.flip(np.sort(x))
    n = len(x)
    p = range(3,n)
    plt.plot(p,[Hill_Estimator(x, k) for k in p],'o')
    plt.show()
    
def VaR_ES_Hill(x, p, k):
    x = np.flip(np.sort(x))
    hillEst = Hill_Estimator(x, k)
    VaR = (((n/k)*(1-p))**(-1/hillEst))*x[k]
    ES = ((1 - 1/hillEst)**(-1))*VaR
    return VaR,ES

if __name__ == '__main__':
    data = np.genfromtxt("RiskMan_2022_Exercise_19_data.dat")
    n = 500
    v = [3,8]
    lmbda = 1
    p = 0.98
    x = [np.random.standard_t(df,n) for df in v]
    e = np.random.exponential(lmbda, n)
    Hill_Plot(x[0], n)
    Hill_Plot(x[1], n)
    Hill_Plot(e, n)
    print(f'The VaR is {VaR_ES_Hill(data, p, 5)[0]} and the ES is {VaR_ES_Hill(data, p, 5)[1]}')

    