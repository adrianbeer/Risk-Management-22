# Rounak Bastola,Adrian Beer
# QF 13

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


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
    VaRs095.append(VaR_log_normal(s.iloc[:i], 0.95))
    VaRs09.append(VaR_log_normal(s.iloc[:i], 0.9))

L = -s.diff().iloc[253:].reset_index(drop=True)
violations09 = L > VaRs09
violations095 = L > VaRs095
assert len(VaRs09) == len(VaRs095) == len(L)


f, ax = plt.subplots(1,1)
ax.plot(L.index, L, color="blue")
ax.plot(L.index, VaRs09, color="green")
ax.plot(L.index, VaRs095, color="violet")
ax.scatter(L.index[violations095], L[violations095], color="red", s=10)
plt.show()


print(f"VaR_0.9%: Violation rate: {violations09.mean():.4f} (expected 0.1)")
print(f"VaR_0.95%: Violation rate: {violations095.mean():.4f} (expected 0.05)")

