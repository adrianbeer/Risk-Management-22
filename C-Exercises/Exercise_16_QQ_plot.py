# QF 13
# Rounak Bastola Adrian Beer

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, t
from functools import partial
import numpy as np
import yfinance as yf

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def qqplot(x, F_inv):
    df = pd.DataFrame(x)
    df.columns = ["Empirical Quantile"]
    df.sort_values(by="Empirical Quantile", ascending=True, inplace=True)
    df["Empirical Percentile"] = (df.reset_index().index + 1) / (df.shape[0]+1)
    df["Theoretical Quantile"] = [F_inv(x) for x in df["Empirical Percentile"].values]

    plt.scatter(x=df["Theoretical Quantile"], y=df["Empirical Quantile"])
    plt.xlabel("Theoretical Quantile")
    plt.ylabel("Empirical Quantile")
    abline(1, 0)
    plt.show()


tickers = ['BMW.DE']
x = yf.download(tickers, start='2000-01-01', end='2022-08-11')
x = x["Close"]
x = x.diff().dropna()

qqplot(x, norm.ppf,)

degfree, loc, scale = t.fit(x)
t_ppf = partial(t.ppf, loc=loc, scale=scale, df=degfree)
qqplot(x, t_ppf)

t_ppf = partial(t.ppf, loc=loc, scale=scale, df=degfree*2)
qqplot(x, t_ppf)

# Comment: For the standard normal distribution the empirical tails are heavier than expected by the distr.
# For the fittted t distribution the tails are lighter than expected by the distr.
