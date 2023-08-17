# Example 11. A study on movie customers has revealed that their
# spending on concessions is approximately normally distributed
# with a mean of $4.11 and a standard deviation of $1.37.
# What percentage of customers will spend more than $3.00 on concessions?

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__ == '__main__':
    mean = 4.11
    std = 1.37
    #x-axis ranges from 0 and 9 with .1 steps
    x = np.arange(0, 9, 0.1)
    #plot cumulative distribution with mean and standard deviation 1
    plt.plot(x, norm.cdf(x, mean, std))
    # naming the x axis
    plt.xlabel('X')
    # naming the y axis
    plt.ylabel('F(X)')
    # giving a title to my graph
    plt.title('normal cumulative density function')
    # save fig and show the plot
    plt.savefig('ex11.png')
    plt.show()
    # output the probability of X > 3
    print(f'P(X>3) = %.4f' % (1 - norm.cdf(3, mean, std)))