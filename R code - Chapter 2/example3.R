# ---
# Example 3.
# An urn contains 10 black and 10 white balls. 
# Draw 3 (a) without replacement, and (b) with replacement. 
# What is the probability that all three are white?
# ---
library(DescTools)

prob_no_repl <- function(){
  # Undered samples of size k, without replacement, from n objects
  nb_samples_no_repl <- CombN(10, 3, repl=FALSE, ord=FALSE)
  nb_total_samples <- CombN(20, 3, repl=FALSE, ord=FALSE)
  return (nb_samples_no_repl / nb_total_samples)
}

prob_with_repl <- function(){
  # Ordered samples of size k, with replacement, from n objects
  nb_samples_with_repl <- CombN(10, 3, repl=TRUE, ord=TRUE)
  nb_total_samples <- CombN(20, 3, repl=TRUE, ord=TRUE)
  return (nb_samples_with_repl / nb_total_samples)
}

p <- prob_no_repl()
q <- prob_with_repl()

cat(paste0("probability of drawing 3 white balls without replacement = ", round(p, 4), "\n"))
cat(paste0("probability of drawing 3 white balls with replacement = ", round(q, 4), "\n"))