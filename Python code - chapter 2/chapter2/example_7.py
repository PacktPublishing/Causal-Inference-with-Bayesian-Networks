# Example 7. A random variable takes on the value
# X = 1 if the outcome of rolling a die is strictly greater than 4
# and X= 0 otherwise.
# What is the probability distribution for X?
# Calculate E[X], Var(X).

from scipy.stats import bernoulli

if __name__ == '__main__':
    # X is a Bernoulli random variable with p=1/3.
    p = 1.0/3
    mean, var = bernoulli.stats(p, moments='mv')
    print(f'E[X] = %.4f' % mean)
    print(f'Var(X) = %.4f' % var)
