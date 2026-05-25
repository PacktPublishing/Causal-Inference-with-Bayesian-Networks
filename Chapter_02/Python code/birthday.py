# Example 2.1 (Birthday Problem).
# Assume that there are k people in the room.
# What is the probability that there are
# two who share a birthday?


from math import perm
import matplotlib.pyplot as plt


def no_shared_birthday_probability(k):
    a = perm(365, k)
    b: int = 365 ** k
    return a / b


def a_shared_birthday_probability(k):
    p = 1 - no_shared_birthday_probability(k)
    return p


def shared_probabilities(list):
    probabilities = []
    for k in list:
        p = a_shared_birthday_probability(k)
        print(k, p)
        probabilities.append(p)
    return probabilities


def make_plot():
    list = [10, 23, 41, 57, 70]
    # plotting the points
    probabilities = shared_probabilities(list)
    plt.plot(list, probabilities)
    # naming the x axis
    plt.xlabel('number of people')
    # naming the y axis
    plt.ylabel('a shared birthday probability')
    # giving a title to my graph
    plt.title('Shared birthday proability')
    plt.savefig('birthday.png')
    # function to show the plot
    plt.show()


if __name__ == '__main__':
    make_plot()
