# Example 9. Let X be the number of customers at a 
# restaurant per day and assume the customer arrival 
# process can be modeled by the Poisson distribution 
# with an average of 100 customers per day. 
# Calculate E[X],Var(X) and P(X>110).
#
lambda <- 100.
cat("E[X] = ", lambda, "\n")
cat("Var(X) = ", lambda, "\n")
cat("P(X>110) = ", 1 - ppois(110., lambda), "\n")

x <- 0:110
plot(dpois(x, lambda), type = "h", lwd = 2,
     main = "Poisson probability mass function",
     ylab = "P(X = x)", xlab = "Number of events")