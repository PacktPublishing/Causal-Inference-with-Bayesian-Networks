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
    p_a =  sum(x * y)
    return (p_a_cond_b[k] * p_b[k]) / p_a


if __name__ == '__main__':
    p_b = [0.13, 0.87]
    p_a_cond_b = [0.92, 0.10]
    # calculate P(A|B[0])
    print(f'P(A|B[0]) %.2f'  % bayes_theorem(0,p_a_cond_b, p_b))
    print(f'P(A|B[1]) %.2f'  % bayes_theorem(1, p_a_cond_b, p_b))
