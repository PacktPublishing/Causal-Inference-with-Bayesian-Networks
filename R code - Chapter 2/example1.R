# ---
# Example 1 (Birthday Problem). 
# Assume that there are k people in a room. 
# What is the probability that there are two who share a birthday? 
# ---
library(DescTools)

no_shared_birthday_probability <- function(k){
  n <- 365  # days in a year
  # Ordered samples of size k, without replacement, from n objects
  a <- CombN(n, k, repl=FALSE, ord=TRUE)
  # Ordered samples of size k, with replacement, from n objects
  b <- CombN(n, k, repl=TRUE, ord=TRUE)
  return (a/b)
}

a_shared_birthday_probability <- function(k){
  p <- 1 - no_shared_birthday_probability(k)
  print(p)
  return (p)
}

make_plot <- function(){
  nb_people <- c(10, 23, 41, 57, 70)
  # plotting the points
  probabilities <- Map(a_shared_birthday_probability, nb_people)
  plot(nb_people, probabilities)
}

make_plot()