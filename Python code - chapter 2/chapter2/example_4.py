# Example 4 Place 3 six-sided dice into a cup.
# Next, shake the cup well and pour out the dice.
# How many distinct rolls are possible?

from math import comb


def count_samples_unordered_with_repl(n, k):
    """
    :param n: population size
    :param k: sample size
    :return: count of number of ways to draw unordered sample of size k with replacement from population of n distinct objects
    """
    return comb(n + k - 1, k)


if __name__ == '__main__':
    print(f'number of distinct rolls of 3 dices = %d' % count_samples_unordered_with_repl(6, 3))
