# Example 9. Let X be the number of customers at a restaurant per day
# and assume the customer arrival process can be modeled by the Poisson
# distribution with an average of 100 customers per day.
# Calculate E[X],Var(X) and P(X>110)

from example_5 import make_plots
from scipy.stats import poisson
import numpy as np


if __name__ == '__main__':
    m = 100
    xk = np.arange(0, 110, 5)
    make_plots(xk, poisson(m), 'ex9.png')
    mean, var = poisson.stats(m, moments='mv')
    print(f'E[X] = %.1f' % mean)
    print(f'Var(X) = %.1f' % var)
    print(f'P(X>110) = %.4f' % (1- poisson(m).cdf(110)))