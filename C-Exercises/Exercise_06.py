# Rounak Bastola,Adrian Beer
# QF 13

import pandas as pd
import numpy as np
from scipy.stats import binom, norm


def _test_binomial(v, p0, beta):
    t = binom.cdf(sum(v), n=len(v), p=p0)
    print(f"cdf-value: {t}")
    if (t < beta/2) or (t > 1-beta/2):
        return 1
    else:
        return 0


def VaR_log_normal(s, alpha):
    X = np.log(1 + s.pct_change())
    mu_hat = X[-251:].mean()
    assert len(X[-251:])
    sigma_hat = np.sqrt(((X[-251:] - mu_hat)**2).sum() / 250)
    VaR = s.iloc[-1] * (1 - np.exp(mu_hat + sigma_hat*norm.ppf(1-alpha)))
    return VaR


df = pd.read_csv("C-Exercises/dax_data.csv", delimiter=';')
s = df.sort_values("Datum").reset_index(drop=True)["Schlusskurs"]

VaRs09 = []
VaRs095 = []
for i in range(252, len(s)-1):
    VaRs09.append(VaR_log_normal(s.iloc[:i], 0.9))
    VaRs095.append(VaR_log_normal(s.iloc[:i], 0.95))

L = -s.diff().iloc[253:].reset_index(drop=True)
violations09 = L > VaRs09
violations095 = L > VaRs095

test2 = _test_binomial(violations09, 0.1, 0.05)
print(test2)

test1 = _test_binomial(violations095, 0.05, 0.05)
print(test1)

# Test rejects for our VaR_095 estimates, not for the VaR_09 estimates.

