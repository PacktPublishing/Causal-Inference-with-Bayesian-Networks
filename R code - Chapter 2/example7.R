# Example 7. A random variable takes on the value X = 1 
# if the outcome of rolling a die is strictly greater than 4 
# and X= 0 otherwise. What is the probability distribution for X? 
# Calculate E[X], Var(X).
#
# Bernoulli(p)
p <- 1./3
cat("E[X] = ", p, "\n")
cat("Var(X) = ", p*(1-p), "\n")
