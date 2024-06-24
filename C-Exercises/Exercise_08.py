# QF 13
# Rounak Bastola, Adrian Beer

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt


def VaR_ES_var_covar(x, c, w, alpha):
    mu_hats = x.apply(np.mean, axis=0)
    #print(mu_hats)
    vcov = np.cov(x.T)

    comb_mu = (c + w.dot(mu_hats))
    comb_var = w.dot(vcov).dot(w)

    var = -comb_mu + np.sqrt(comb_var) * norm.ppf(q=alpha, loc=0, scale=1)
    es = -comb_mu + np.sqrt(comb_var) * norm.pdf(norm.ppf(alpha))/(1-alpha)

    return var, es


tickers = ['BMW.DE', 'SIE.DE', 'SAP.DE', 'CON.DE', 'VOW.DE']
df = yf.download(tickers, start='2000-01-01', end='2022-08-11')
df = df.xs("Close", level="Price", drop_level=True, axis=1)
x = df. apply(lambda x: np.log(1 + x.pct_change().dropna()))

w = np.array([38, 31, 24,  50, 22])*df


VaRs = []
ESs = []
for i in range(252, len(x)):
    var, es = VaR_ES_var_covar(x.iloc[i-252:i, :], c=0, w=w.iloc[i-1], alpha=0.98)
    VaRs.append(var)
    ESs.append(es)

x_grid = df.index[252:-1]
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x_grid, VaRs, label="VaR")
ax1.plot(x_grid, ESs, label="ES")
ax1.set_xlabel("Time")
ax1.legend()

ax2.plot(x_grid, np.array(ESs)-np.array(VaRs), label="ES - VaR")
ax2.legend()
plt.show()

# Comment: ES and VaR highly correlated.

