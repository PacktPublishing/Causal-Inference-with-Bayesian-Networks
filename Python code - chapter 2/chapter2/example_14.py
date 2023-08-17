# Example 14. Assume a doctor believes there is a 30% chance
# the patient has a certain illness. The doctor orders a diagnostic test.
# The test returns a positive result 95% of the time for patients
# who have that illness. The test returns a positive result 7% of
# the time for people who do not have that illness.
# Assume the test came positive,
# what is the posterior probability the patient has that illness?

from bayes import bayes_theorem

if __name__ == '__main__':
    p_b = [0.3, 0.7]
    p_a_cond_b = [0.95, 0.07]
    # calculate P(A|B[0])
    print(f'P(A|B[0]) %.2f' % bayes_theorem(0, p_a_cond_b, p_b))
    print(f'P(A|B[1]) %.2f' % bayes_theorem(1, p_a_cond_b, p_b))
