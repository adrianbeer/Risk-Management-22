# Sample solution C15, WT2022

import math
import numpy as np
from scipy import optimize

# part a)
def log_likelihood_ARCH2(theta, x):
    # number of observations
    n = len(x)

    # compute sigma values from x
    sigma_sq = np.ones(n - 2)
    for i in range(2, n):
        sigma_sq[i - 2] = theta[0] + theta[1] * x[i - 1] ** 2 + theta[2] * x[i - 2] ** 2

    # log likelihood
    y = -0.5 * (n * math.log(2 * math.pi) + np.sum(np.log(sigma_sq)))
    y -= 0.5 * np.sum(np.divide(np.power(x[2:], 2), sigma_sq))

    return y


# for the optimization we need the negative log likelihood function
def neg_log_likelihood_ARCH_2(theta, x):
    y = log_likelihood_ARCH2(theta, x)
    return -y


# part b)
def estimates_ARCH2(x):
    # initial theta parameters
    theta_0 = np.array([0.01, 0.2, 0.8])

    # optimization command
    theta_hat = optimize.minimize(neg_log_likelihood_ARCH_2, args=x, x0=theta_0,
                                         bounds=((0.0001, 5), (0.00001, 5), (0.00001, 5)))
    return theta_hat.x


# part c)
def VaR_ES_ARCH2_MC(k, m, l, alpha, x):
    n = len(x)
    theta = estimates_ARCH2(x)
    # simulation for X: m is the number of paths, k the length of the paths
    X = np.zeros((m, k+2))
    Y = np.random.normal(size=(m, k))
    # use last values of our data to start the simulation
    X[:, 0] = x[n - 2]
    X[:, 1] = x[n - 1]
    for j in range(0, k):
        sigma = theta[0] + theta[1] * np.power(X[:, j], 2) + theta[2] * np.power(X[:, j + 1], 2)
        X[:, j + 2] = np.multiply(np.sqrt(sigma), Y[:, j])

    # losses
    loss = np.ones(m)
    for j in range(0, m):
        loss[j] = l(X[j, :])

    # Value at Risk and Expected Shortfall
    l_data_sorted = np.flip(np.sort(loss))

    VaR = l_data_sorted[int(np.floor(m * (1 - alpha)) + 1)]
    ES = 1 / (np.floor(m * (1 - alpha)) + 1) * np.sum(l_data_sorted[0:int(np.floor(m * (1 - alpha)) + 1)])
    return VaR, ES


# part d)

# application to DAX time series
# level Value at Risk
alpha = 0.98

# number of simulations in Monte-Carlo method
m = 1000

# forecast period
k = 5

# load DAX time series
dax = np.flip(np.genfromtxt('C-Exercises/dax_data.csv', delimiter=';', skip_header=1, usecols=4))
# compute log returns
x = np.diff(np.log(dax))

# loss operator
def l(x):
    return dax[-1] * (1 - np.exp(np.sum(x)))

# parameter estimation
theta_hat = estimates_ARCH2(x)
print('Estimated parameters for theta and sigma1: ' + str(theta_hat))

# compute VaR and ES
VaR, ES = VaR_ES_ARCH2_MC(k, m, l, alpha, x)

# display the result
print(str(k) + '-day ahead MC estimates:')
print('    VaR: ' + str(VaR) + '          Es: ' + str(ES))
