# ---
# rv_discrete.R
# Example 3: Consider the sample space of tossing a fair coin three times.
# The random variable X gives the number of heads recorded. Compute the
# probability mass function and the cumulative distribution function for X.
# ---

probability <- function() {
  count <- c(0, 0, 0, 0)
  outcomes <- c(0, 1)
  for (i in outcomes) {
    for (j in outcomes) {
      for (k in outcomes) {
        lst <- c(i, j, k)
        s <- sum(lst)
        if (s == 0) count[1] <- count[1] + 1
        else if (s == 1) count[2] <- count[2] + 1
        else if (s == 2) count[3] <- count[3] + 1
        else if (s == 3) count[4] <- count[4] + 1
      }
    }
  }
  return(count / sum(count))
}

pmf = probability()
cdf <- c(pmf[1])
for (i in 2:length(pmf)) {
  cdf[i] <- cdf[i-1]+pmf[i]
}
print(pmf)
print(cdf)