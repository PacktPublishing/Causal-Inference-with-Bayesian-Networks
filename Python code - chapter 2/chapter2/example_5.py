# Example 5: Consider the sample space of tossing a fair coin three times.
# The random variable X gives the number of heads recorded.
# Compute the probability mass function for X

import matplotlib.pyplot as plt
from scipy import stats
"""
plot the probability mass function (PMF) and the cumulative
distribution function (CDF) for custom discrete random variable (cust)
and save figures in filename
"""

def make_plots(xk,custm,filename):
    """
    :param xk: array of integers
    :param custm: custom discrete random variable
    :param filename: file name to save the figure tp
    :return: probability mass function (PMF) and cumulative distribution function (CDF) plots
    """
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
    ax[0].vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
    ax[0].set_ylabel('PMF')
    ax[1].plot(xk, custm.cdf(xk), 'ro', ms=12, mec='r')
    ax[1].vlines(xk, 0, custm.cdf(xk), colors='r', lw=4)
    ax[1].set_ylabel('CDF')
    plt.show()
    fig.savefig(filename)


# Example 5-
# Consider the sample space of tossing a fair coin three times.
# The random variable X gives the number of heads recorded.
# Compute the probability mass function and the
# cumulative distribution function for X.
#
# The possible values of X are 0,1,2,3.
# The sample space is {HHH,HHT,HTH,HTT,THH,THT,TTH,TTT},
# and each outcome is equally likely.
# The event X = 1, for example, when written as a set of outcomes,
# is equal to {HTT,THT,TTH}, and has probability 3/8.
# The PMF is given by the following table:
#
# x   P(X=x)
#
# 0       1/8
# 1       3/8
# 2       3/8
# 3       1/8

if __name__ == '__main__':
    xk = [0, 1, 2, 3]
    pk = [1/8, 3/8, 3/8, 1/8]
    custm = stats.rv_discrete(name='custm', values=(xk, pk))
    make_plots(xk, custm, 'ex5.png')
