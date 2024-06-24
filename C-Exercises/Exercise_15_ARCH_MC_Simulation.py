#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:15:51 2022

@author: me
"""
import numpy as np
import pandas_datareader as pdr
from scipy.optimize import minimize


def log_likelihood_ARCH2(theta,x):
    sigma = np.zeros(len(x))
    l = np.zeros(len(x))
    sigma[0] = theta[3]
    l[0] = .5*np.log(sigma[0]) + (x[0]**2)/sigma[0]
    sigma[1] = theta[0] + theta[1]*(x[0]**2)
    l[1] = .5*np.log(sigma[1]) + (x[1]**2)/sigma[1]
    for n in range(2,len(x)):
        sigma[n] = theta[0] + theta[1]*(x[n-1]**2) + theta[2]*(x[n-2]**2)
        l = .5*np.log(sigma[n]) + (x[n]**2)/sigma[n]
    return (n/2)*np.log(2*np.pi)+np.sum(l)


def estimates_ARCH2(x):
    return minimize(log_likelihood_ARCH2, x0=[0.01,0.2,0.8,np.var(x)],args=(x),bounds=[(1e-5,5)]*4, options={"disp":True}).x

def VaR_ES_ARCH2_MC(k,m,l,alpha,x):
    theta = estimates_ARCH2(l(x))
    x = l(x[-k-1:-1])
    sim = np.random.normal(0,1,size=(m,k))
    sigma = np.zeros((m,k))
    sigma[:,0] = np.var(x)
    sim[:,0] *= np.sqrt(sigma[:,0])
    sigma[:,1] = theta[0] + theta[1]*(sim[:,0]**2)
    sim[:,1] *= np.sqrt(sigma[:,1])
    for n in range(2,k):
        sigma[:,n] = theta[0] + theta[1]*(sim[:,n-1]**2) + theta[2]*(sim[:,n-2]**2)
        sim[:,n]*= np.sqrt(sigma[:,n])
    sim_sort = np.flip(np.sort(sim[:,-1]))
    index = int(np.floor(m*(1-alpha)))+1
    VaR = np.mean(sim_sort[index])
    ES = np.mean((1/(index))*np.sum(sim_sort[:index]))
    return VaR, ES
    
        
    



if __name__ == '__main__':
    tickers = ['^GDAXI']
    df = pdr.DataReader(tickers, data_source='yahoo', start='1990-01-01', end='2022-10-21')
    df = df.xs("Close", level="Attributes", drop_level=True, axis=1)
    data = np.array(df).T[0]
    x = np.diff(np.log(data))
    l  = lambda x: data[-1]*(1-np.exp(x))
    k =5 
    m =1000
    alpha = 0.98
    VaR_ES = VaR_ES_ARCH2_MC(k, m, l, alpha, x)
    print(f'The VaR is {VaR_ES[0]} and the ES is {VaR_ES[1]}')

    