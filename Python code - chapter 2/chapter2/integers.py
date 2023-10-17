# Example 2.2 [Non-decreasing integers]
# If we are to sample with replacement k digits in the range 0 to 9,
# what is the probability that the sequence of k digits is in non-decreasing
# order? Calculate the probability for k=2 to 10.
from math import comb

def count_samples_unordered_with_repl(n, k):
    return comb(n + k - 1, k)

def nd_integers(k):
    return count_samples_unordered_with_repl(10, k)


if __name__ == '__main__':
    probability = {}
    for k in range(2, 10) :
        probability[k] = nd_integers(k)/(10**k)
    print(probability)