# ---
# Example 6.
# An urn contains 20 balls numbered 1,⋯,20. 
# Select 5 balls at random, without replacement. 
# Let X be the largest number among selected balls. 
# Determine its PMF and the probability that at 
# least one of the selected numbers is 15 or more. 
# ---
library(DescTools)

count_samples_unordered_without_repl <- function(n, k){
  # n: population size
  # k: sample size
  # return: count of number of ways to draw unordered sample of size k with replacement from population of n distinct objects
  CombN(n, k, repl=FALSE, ord=FALSE)
}

prob <- function(x){
  a <- choose(x-1,4) #count_samples_unordered_without_repl(x-1,4)
  b <- choose(20,5) #count_samples_unordered_without_repl(20, 5)
  return(a/b)
}

x <- c()
pmf <- c()
cdf <- c()

for (i in 5:20){
  x <- c(x, i)
  pmf <- c(pmf, prob(i))
  cdf <- c(cdf, Reduce("+",pmf))
}
print(pmf)
print(cdf)
print(paste0('P(X>=15) = ', 1 - cdf[10]))