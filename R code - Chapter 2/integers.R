# ---
# integers.R
# ---
# integers.R
# What is the number of outcomes of constructing an integer of k digits
# such that the digits take values in the range 0 to 9, are in
# non-decreasing order, and repeated digits are allowed?
# Calculate the count for k = 1 to 20. If we are to sample with
# replacement k digits in the range 0 to 9, what is the probability
# that the sequence of k digits is in non-decreasing order?
# Calculate the probability for k=2 to 10.
# ---
library(DescTools)

nd_integers <- function(k){
  # number of ways to draw non-decreasing digits of size k
  CombN(10, k, repl=TRUE, ord=FALSE)
}

for (k in 2:10) {
  p = nd_integers(k)/(10^k)
  cat("probability for k=", k, "is",  p,"\n")
}
