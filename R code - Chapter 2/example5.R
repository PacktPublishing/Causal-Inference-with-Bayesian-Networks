# ---
# Example 5.
# Consider the sample space of tossing a fair coin three times. 
# The random variable X gives the number of heads recorded. 
# Compute the probability mass function and the cumulative 
# distribution function for X.
# ---
x=c(0, 1, 2, 3)
pmf=c(1/8, 3/8, 3/8, 1/8)
cdf <- c(1/8)
for (i in 2:length(pmf)) {
  cdf[i] <- cdf[i-1]+pmf[i]
}
print(pmf)
print(cdf)

