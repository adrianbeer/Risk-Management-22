# QF 13
# Rounak Bastola
# Adrian Beer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm, expon
from scipy.optimize import minimize

def MEF(x, u):
    assert u < max(x)
    mean_excess = (x[x > u] - u).mean()
    return mean_excess


def MEP(x):
    mes = []
    x = np.sort(x)
    for u in x[:-1]:
        mes.append(MEF(x, u))
    plt.scatter(x=x[:-1], y=mes)
    plt.xlabel("u")
    plt.ylabel("Mean Excess")


t3_sims = t.rvs(df=3, size=500)
MEP(t3_sims)
plt.title("t_3 distribution")
plt.show()

t8_sims = t.rvs(df=8, size=500)
MEP(t8_sims)
plt.title("t_8 distribution")
plt.show()

exp_sims = expon.rvs(loc=1, size=500)
MEP(exp_sims)
plt.title("expon_1 distribution")
plt.show()

def PoT_estimated(x, u):
    y = x[x > u] - u # excesses
    N_u = len(y)
    def neg_ll(theta):
        beta, gamma = theta
        return + N_u * np.log(beta) - (1/gamma +1) * np.log(1 + gamma/beta*y).sum()
    res = minimize(neg_ll, x0=(1,1), bounds=((0.01, np.inf), (0.01, np.inf)), method="Nelder-Mead", options={"disp":True})
    beta_hat, gamma_hat = res.x
    return beta_hat, gamma_hat


def VaR_ES_PoT(x, p, u):
    n = len(x)
    N_u = len(x[x > u])
    beta, gamma = PoT_estimated(x, u)
    var = u + beta * ((n/N_u*(1-p))**(-gamma)-1)
    es = var + (beta + gamma*(var - u))/(1 - gamma)
    return var, es


data = np.genfromtxt("RiskMan_2022_Exercise_19_data.dat")
MEP(data)  # Doesn't really become linear anywhere... chose u=10?
plt.title("data")
plt.show()

print(VaR_ES_PoT(data, 0.98, 4))
