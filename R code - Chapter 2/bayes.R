# Example 2.6:
# Assume a doctor believes there is a
# 30% chance the patient has a certain illness. 
# The doctor orders a diagnostic test. 
# The test returns a positive result 95% of the 
# time for patients who have that illness. 
# The test returns a positive result 7% of the 
# time for people who do not have that illness. 
# Assume the test came positive, 
# what is the posterior probability the patient has that illness?

bayes_theorem <- function(k, p_a_cond_b, p_b) {
  x <- as.numeric(p_a_cond_b)
  y <- as.numeric(p_b)
  
  p_a <- sum(x * y)
  return((p_a_cond_b[k] * p_b[k]) / p_a)
}

p_b <- c(0.3, 0.7)
p_a_cond_b <- c(0.95, 0.07)

cat(sprintf("P(A|B[1]) %.3f\n", bayes_theorem(1, p_a_cond_b, p_b)))
cat(sprintf("P(A|B[2]) %.3f\n", bayes_theorem(2, p_a_cond_b, p_b)))