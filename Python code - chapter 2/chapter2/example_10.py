# Example 10.  A game consists of one person A roll a die
# and an opponent B tosses a coin.
# If A rolls a 6 then A wins and if A does not roll a 6
# and B tosses Heads then A loses; otherwise,
# the game continues another round.
# On average, how many rounds does the game last

from example_5 import make_plots
import numpy as np
from scipy.stats import geom

if __name__ == '__main__':
    p = 7.0/12.0
    xk = np.arange(0,5)
    make_plots(xk,geom(p),'ex10.png')
    X = geom(p)
    print(f'E[X] = %.1f' % X.mean())