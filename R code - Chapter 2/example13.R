# Example 13. A box has a two-headed coin and a fair coin. 
# A coin is picked up and flipped n times, yielding heads 
# each time. 
# What is the probability that the two-headed coin is chosen?

bayes_theorem <- function(k, p_a_cond_b, p_b) {
  x <- as.array(p_a_cond_b)
  y <- as.array(p_b)
  
  p_a <- sum(x * y)
  return((p_a_cond_b[k] * p_b[k]) / p_a)
}

cond_prob <- function(n) {
  return(c(1, (1 / 2 ^ n)))
}

post_prob <- function(n) {
  p_b <- c(1 / 2, 1 / 2)
  p_a_cond_b <- c(1, (1 / 2 ^ n))
  
  return(bayes_theorem(1, p_a_cond_b, p_b))
}

make_plot <- function() {
  x <- c()
  y <- c()
  for (n in 0:9) {
    x <- c(x, n)
    y <- c(y, post_prob(n))
  }
  plot(x, y, type = "l", xlab = "n tosses all heads", ylab = "probability of two-headed coin")
  dev.copy(png, "ex13.png")
  dev.off()
}

make_plot()