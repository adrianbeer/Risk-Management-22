# QF 13
# Adrian BEer Rounak Bastola
from matplotlib import pyplot as plt
from scipy.stats import norm, t, expon
import numpy as np

np.random.seed(42069)


def Hill_Estimator(x, k):
    Y = np.sort(x)
    est = k / (np.log(Y[-(k-1):]) - np.log(Y[-k])).sum()
    return est


def Hill_Plot(x):
    k_grid = list(range(3, len(x)))
    estimates = [Hill_Estimator(x, k) for k in k_grid]
    plt.scatter(x=k_grid, y=estimates)
    plt.show()


t3_sims = t.rvs(df=3, size=500)
Hill_Plot(t3_sims)
# plot => Choose alpha ~ 3

t8_sims = t.rvs(df=8, size=500)
Hill_Plot(t8_sims)
# hopeless, no estimate

exp_sims = expon.rvs(loc=1, size=500)
Hill_Plot(exp_sims)
# plot => alpha ~ 6.5


def VaR_ES_Hill(x, p, k):
    x = np.sort(x)
    n = x.shape[0]
    hill_alpha = Hill_Estimator(x, k)
    var_est = (n/k*(1-p))**(-1/hill_alpha)*x[-k]
    es_est = var_est / (1-1/hill_alpha)
    return var_est, es_est

# Dataset
data = np.genfromtxt("C-Exercises/RiskMan_2022_Exercise_19_data.dat")
Hill_Plot(data) # Choose alpha = 8 or k=10 based on hill plot
print(VaR_ES_Hill(data, 0.98, 8))