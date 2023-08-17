# Example 8. Let X be the number of Heads in 50 tosses of a fair coin.
# Calculate E[X],Var(X) and P(X≤10).
from example_5 import make_plots
from scipy.stats import binom
import numpy as np


if __name__ == '__main__':
    n=50
    p=0.5
    xk = np.arange(0,50)
    make_plots(xk,binom(n,p),'ex8.png')
    X = binom(n,p)
    print(f'E[X] = %.4f' % X.mean())
    print(f'Var(X) = %.4f' % X.var())
    print(f'P(X=<20) = %.4f' % X.cdf(20))