# Example 13. A box has a two-headed coin and a fair coin.
# A coin is picked up and flipped n times, yielding heads each time.
# What is the probability that the two-headed coin is chosen?

import matplotlib.pyplot as plt
from bayes import bayes_theorem

# conditional probability of getting n heads
# given the coin is two-headed or fair is
def cond_prob(n):
    return [1., (1./2 ** n)]

# posterior probability for the T hypothesis
def post_prob(n):
    # prior probability of the two hypotheses
    # Two_headed (T) or Fair (F)
    p_b = [1. / 2, 1. / 2]
    # conditional probability of getting n heads
    # given each hypothesis
    p_a_cond_b = [1., (1./2 ** n)]
    # posterior for first hypothesis
    return bayes_theorem(0,p_a_cond_b, p_b)

def make_plot():
    x = []
    y = []
    for n in range(10):
        x.append(n)
        y.append(post_prob(n))
    plt.plot(x, y)
    plt.xlabel('n tosses all heads')
    plt.ylabel('probability of two-headed coin')
    plt.savefig('ex13.png')
    plt.show()

if __name__ == '__main__':
    make_plot()