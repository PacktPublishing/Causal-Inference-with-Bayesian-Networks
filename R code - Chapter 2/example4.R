# ---
# Example 4.
# Place 3 six-sided dice into a cup. Next, shake the cup well 
# and pour out the dice. How many distinct rolls are possible? 
# ---
library(DescTools)

count_samples_unordered_without_repl <- function(n, k){
  # n: population size
  # k: sample size
  # return: count of number of ways to draw unordered sample of size k with replacement from population of n distinct objects
  CombN(n, k, repl=TRUE, ord=FALSE)
}

cat(paste0("number of distinct rolls of 3 dices = ", count_samples_unordered_with_repl(6, 3)))