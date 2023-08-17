# Example 10.  A game consists of one person 
# A roll a die and an opponent B tosses a coin. 
# If A rolls a 6 then A wins and if A does not 
# roll a 6 and B tosses Heads then A loses; 
# otherwise, the game continues another round. 
# On average, how many rounds does the game last?

# the number of rounds is dgeom(7./12)
p <- 7./12
cat("E[X] = ", 1/p, "\n")

# make plot
# Grid of X-axis values
x <- 0:5
plot(dgeom(x, prob = p), type = "h", lwd = 2,
     main = "Geometric probability mass function",
     ylab = "P(X = x)", xlab = "Number of rounds")