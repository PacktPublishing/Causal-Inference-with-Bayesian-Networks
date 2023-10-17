# Example 2.3:
# Consider the sample space of tossing a fair coin three times.
# The random variable X gives the number of heads recorded. Compute the
# probability mass function and the cumulative distribution function for X.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
plot the probability mass function (PMF) and the cumulative
distribution function (CDF) for custom discrete random variable (cust)
and save figures in filename
"""


def make_plots(xk, custm, filename):
    """
    :param xk: array of integers
    :param custm: custom discrete random variable
    :param filename: file name to save the figure to
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


def probability():
    count = [0, 0, 0, 0]
    outcomes = [0, 1]
    for i in outcomes:
        for j in outcomes:
            for k in outcomes:
                lst = [i, j, k]
                s = sum(lst)
                if s == 0:
                    count[0] += 1
                elif s == 1:
                    count[1] += 1
                elif s == 2:
                    count[2] += 1
                elif s == 3:
                    count[3] += 1
                print(lst, s)
    print(count)
    probability = np.divide(count, sum(count))
    print(probability)
    return probability


if __name__ == '__main__':
    pk = probability()
    xk = [0, 1, 2, 3]
    custm = stats.rv_discrete(name='custm', values=(xk, pk))
    make_plots(xk, custm, 'rv_discrete.png')
