# ---
# Example 2. 
# We have a bag that contains 100 balls, 50 of them red and 50 blue. 
# Select 5 balls at random. What is the probability that 3 are blue 
# and 2 are red?
# ---
prob <- function(){
  blue_samples <- choose(50, 3)
  red_samples <- choose(50, 2)
  total_samples <- choose(100, 5)
  probability <- blue_samples * red_samples / total_samples
  print(paste0("probability of drawing 5 balls 3 blue and 2 red = ", round(probability, 4)))
  return(probability)
}

prob()