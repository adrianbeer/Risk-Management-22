# Adrian Beer
# QF 13

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


mu = 0.0002681
sigma = 0.0140599

### a)
daily_log_returns = np.random.normal(loc=mu, scale=sigma, size=8368-1)
plt.plot(daily_log_returns)
plt.xlabel("Time")
plt.ylabel("Daily Log-Return")
plt.show()

### b)
s1 = 1790.37
S = s1 * np.cumprod(np.exp(daily_log_returns))
plt.plot(S)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()
