# Sample Solution C-Exercise 12, WT 2022

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from C_Ex10 import VaR_ES_historic


# part a)

# Generating a random sample, sorting it and plotting the empirical cdf for this sample
# Return X is the random sample, because we need it in part b)
def empirical_cdf(n):
    X = np.random.normal(0, 1, n)

    X_sorted = np.sort(X)

    # Since the empirical cdf is 0 before the first observation and 1 after the last observation,
    # we add a starting point and an end point to our vector (for plotting).
    # Keep in mind, that this is not part of our sample and should not be part of the returned values.
    X_hat = np.zeros(n + 2)
    X_hat[0] = X_sorted[0] - 2
    X_hat[-1] = X_sorted[-1] + 2
    X_hat[1:-1] = X_sorted

    # Taking the i-th and i+1-th entries of the vector and plotting them against (i-1)*1/n
    # For higher number iterations this is quite slow, but the plot does not contain the vertical lines
    for i in range(0, len(X_hat) - 1):
        plt.plot(X_hat[i:i + 2], np.array([i / n, i / n]), 'blue')

    return X


if __name__ == "__main__":
    # the range for our normal distribution
    x = np.linspace(-4, 4, 1000)
    # plots
    X_10 = empirical_cdf(10)
    plt.plot(x, stats.norm.cdf(x), 'red')
    plt.title('Empirical cdf of N(0,1) with sample size n = 10 vs theoretical cdf')
    plt.figure()
    X_100 = empirical_cdf(100)
    plt.plot(x, stats.norm.cdf(x), 'red')
    plt.title('Empirical cdf of N(0,1) with sample size n = 100 vs theoretical cdf')
    plt.figure()
    X_1000 = empirical_cdf(1000)
    plt.plot(x, stats.norm.cdf(x), 'red')
    plt.title('Empirical cdf of N(0,1) with sample size n = 1000 vs theoretical cdf')

    # part b)
    alpha = np.array([0.9, 0.975])
    m = len(alpha)


    # define loss operator and compute VaR and ES with historical simulation
    # Since we have L_{n+1} ~ N(0,1) our loss operator is just the identity.
    def l(x):
        return x


    VaR = np.zeros(4 * m)
    ES = np.zeros(4 * m)

    for j in range(0, m):
        VaR[j] = stats.norm.ppf(alpha[j])
        ES[j] = stats.norm.pdf(VaR[j]) / (1 - alpha[j])
        VaR[j + m], ES[j + m] = VaR_ES_historic(X_10, l, alpha[j])
        VaR[j + 2 * m], ES[j + 2 * m] = VaR_ES_historic(X_100, l, alpha[j])
        VaR[j + 3 * m], ES[j + 3 * m] = VaR_ES_historic(X_1000, l, alpha[j])

    # compare empirical and theoretical values
    # as one can see for higher n the empirical value converges to the theoretical
    for j in range(0, m):
        print('For alpha = ' + str(alpha[j] * 100) + '%:')
        print('The theoretical VaR is ' + str(VaR[j]) + ' and the theoretical ES is ' + str(ES[j]))
        print('Using the method of historic simulation to estimate the VaR and ES we observe: ')
        print('for 10 observations: VaR is ' + str(VaR[j + m]) + ' , ES is ' + str(ES[j + m]))
        print('for 100 observations: VaR is ' + str(VaR[j + 2 * m]) + ' , ES is ' + str(ES[j + 2 * m]))
        print('for 1000 observations: VaR is ' + str(VaR[j + 3 * m]) + ' , ES is ' + str(ES[j + 3 * m]))

    plt.show()
