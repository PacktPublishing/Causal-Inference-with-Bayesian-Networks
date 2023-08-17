# Example 6. An urn contains 20 balls numbered 1,⋯,20.
# Select 5 balls at random, without replacement.
# Let X be the largest number among selected balls.
# Determine its PMF and the probability that at least
# one of the selected numbers is 15 or more.
from example_5 import make_plots

from math import comb
from scipy.stats import rv_discrete

def f(x):
    return comb(x - 1, 4) / comb(20, 5)

if __name__ == '__main__':
    xk = []
    pk = []
    for i in range(5,21):
        xk.append(i)
        pk.append(f(i))
    custm = rv_discrete(name='custm', values=(xk, pk))
    make_plots(xk, custm, 'ex6.png')
    print(f'P(X>=15) = %.4f' % (1 - custm.cdf(14)))