# Example 3
# An urn contains 10 black and 10 white balls. Draw 3
# (a) without replacement, and
# (b) with replacement.
# What is the probability that all three are white?
from math import comb


# number of samples drawing 3 white out of 10 without replacement
def prob_no_repl():
    nb_samples_no_repl = comb(10, 3)
    nb_total_samples = comb(20, 3)
    return nb_samples_no_repl / nb_total_samples


#
# number of samples drawing 3 white out of 10 with replacement
def prob_with_repl():
    nb_samples_with_repl = 10 ** 3
    nb_total_samples = 20 ** 3
    return nb_samples_with_repl / nb_total_samples


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p = prob_no_repl()
    q = prob_with_repl()
    print(f'probability of drawing 3 white balls without replacement = %.4f' % p)
    print(f'probability of drawing 3 white balls with replacement = %.4f' % q)
