# Example 2.
# We have a bag that contains 100 balls, 50 of them red and 50 blue.
# Select 5 balls at random.
# What is the probability that 3 are blue and 2 are red?

from math import comb

def prob():
    blue_samples = comb(50, 3)
    red_samples = comb(50, 2)
    total_samples = comb(100, 5)
    probability = blue_samples * red_samples / total_samples
    print(f'probability of drawing 5 balls 3 blue and 2 red = %.4f' % probability)
    return probability

if __name__ == '__main__':
    prob()
