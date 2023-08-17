# Example 12. Suppose we have 3 cards identical in form except 
# that both sides of the first card are colored red, both sides 
# of the second card are colored black, and one side of the 
# third card is colored red, and the other side is colored black. 
# The 3 cards are mixed up in a hat, and 1 card is randomly 
# selected and put down on the ground. 
# If the upper side of the chosen card is colored red, what is 
# the probability that the other side is colored black?

bayes_theorem <- function(k, p_a_cond_b, p_b){
  x <- as.vector(p_a_cond_b)
  y <- as.vector(p_b)
  # calculate P(A)
  p_a <- sum(x * y)
  return (p_a_cond_b[k] * p_b[k]) / p_a
}

p_b <- c(1/3, 1/3, 1/3)
p_a_cond_b <- c(1, 1/2, 0)

print(paste0("P(other side is black|upper side is red) ", bayes_theorem(1, p_a_cond_b, p_b)))