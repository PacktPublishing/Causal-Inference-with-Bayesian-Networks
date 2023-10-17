# Example 2.6 [Belief revision]
# Assume a doctor believes there is a 30% chance the patient has a specific illness.
# The doctor orders a diagnostic test. The test returns a positive result 95% of the
# ime for patients with that illness. The test yields a positive result 7% of the
# time for people who do not have that illness. Assuming the test came positive,
# what is the posterior probability the patient has that illness?

import numpy as np


def bayes_theorem(k, p_a_cond_b, p_b):
    """
    :param k: hypothesis index between 0 and n-1, where n is the number of hypotheses
    :param p_a_cond_b: probability of A conditioned on the n hypotheses [P(A|B1),..,P(A|Bn)]
    :param p_b: probabilities of n hypotheses [P(B1),...,P(Bn)
    :return: the posterior probability of the kth hypothesis given A
    """
    x = np.array(p_a_cond_b)
    y = np.array(p_b)
    # calculate P(A)
    p_a = sum(x * y)
    return (p_a_cond_b[k] * p_b[k]) / p_a


if __name__ == '__main__':
    p_b = [0.3, 0.7]
    p_a_cond_b = [0.95, 0.07]
    # calculate P(A|B[0])
    print(f'P(A|B[0]) %.3f' % bayes_theorem(0, p_a_cond_b, p_b))
    print(f'P(A|B[1]) %.3f' % bayes_theorem(1, p_a_cond_b, p_b))
