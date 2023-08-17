# Example 11. A study on movie customers has revealed that 
# their spending on concessions is approximately normally 
# distributed with a mean of $4.11 and a standard deviation 
# of $1.37. What percentage of customers will spend more 
# than $3.00 on concessions?
#
mean <- 4.11
sd <- 1.37
# pnorm is the cumulative distribution
p <- pnorm(3, mean, sd)
cat("P(X>3)", 1-p, "\n")

# plot the normal denisty function
x <- seq(0, 9, 0.1)
plot(x, dnorm(x, mean = mean, sd = sd), type = "l",
     ylim = c(0, 0.3), ylab = "f(x)", lwd = 2, col = "red",
     main = "Normal distributon")