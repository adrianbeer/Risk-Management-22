# Adrian Beer
# QF 13

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

import matplotlib

###  a)
df = pd.read_csv("C-Exercises/dax_data.csv", delimiter=';')
closes = df.sort_values("Datum").reset_index(drop=True)["Schlusskurs"]

print(closes.head())

### b)
log_returns = np.log(1 + closes.pct_change())
log_returns.dropna(inplace=True)

### c)
fig, ax = plt.subplots(1,1)
ax.hist(log_returns, bins=30, density=True, label="Histogram")

### d)
N = len(log_returns)
mu_hat = log_returns.mean()
sigma_hat = np.sqrt(((log_returns - mu_hat)**2/(N-1)).sum())
print(f"mu_hat: {mu_hat:.4f}")
print(f"sigma_hat: {sigma_hat:.4f}")

### e)
x_grid = np.arange(start=-0.1, stop=0.1, step=0.2/60)
densities = norm.pdf(x_grid, loc=mu_hat, scale=sigma_hat)
ax.plot(x_grid, densities, label="Fitted Gaussian")
ax.legend()
ax.set_ylabel("Density")
ax.set_xlabel("Daily Log-Return")
plt.show()
