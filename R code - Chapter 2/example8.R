# Example 8. Let X be the number of Heads in 50 tosses of a 
# fair coin. What is the probability distribution for X? 
# Calculate, E[X],Var(X) and P(X≤20)
#
n <- 50
p <- 0.5

cat("E[X] = ", n*p, "\n")
cat("Var(X) = ", n*p*(1-p), "\n")
# The probability of X=<20, 
cat("P(X=<20) = ",pbinom(20, size = n, prob = p) , "\n")

# make plot
# Grid of X-axis values
x <- 0:n
plot(dbinom(x, size = n, prob = p), type = "h", lwd = 2,
     main = "Binomial probability mass function",
     ylab = "P(X = x)", xlab = "Number of successes")
