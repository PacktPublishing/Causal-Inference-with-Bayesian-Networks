# Example 12. Suppose we have 3 cards identical in form except
# that both sides of the first card are colored red, both sides
# of the second card are colored black, and one side of the third
# card is colored red, and the other side is colored black.
# The 3 cards are mixed up in a hat, and 1 card is randomly selected
# and put down on the ground.
# If the upper side of the chosen card is colored red,
# what is the probability that the other side is colored black?

from bayes import bayes_theorem

if __name__ == '__main__':
    # probability of 3 hypotheses RR, RB, BB
    p_b = [1./3, 1./3, 1./3]
    # probability upper side red conditioning on
    # each hypotheses R|RR, R|RB, R|BB
    p_a_cond_b = [1., 1./2, 0]
    # calculate posterior probability for second hypothesis
    # P(A|B[1])
    print(f'P(other side is black|upper side is red) %.2f' % bayes_theorem(1, p_a_cond_b, p_b))

