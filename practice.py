# at each point we need to calculate the slope
# learning rate can be used in conjuction with the slope to determine the next point
# calculus helps figuring out these baby steps
import numpy as np
import pandas as pd

# derivitive is all about slopes
# slope = change in y/ change in x
# at each step cost function needs to be reduced

data = pd.read_csv('data/stat_females.csv', delimiter='\t').to_numpy()


# bias and weights??
# epoch is a full pass through the data features
# 4 features= 4 weights updates per epoch

x = data[:, 1]
y = data[:, 2]

def grad_descent(x, y):
    # start with some value of m an b
    # take the baby steps towards the min
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.08
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = 1/n * sum([val ** 2 for val in (y-y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}  iteration {}".format(m_curr, b_curr, cost, i))

#x = np.array([1, 2, 3, 4, 5])
#y = np.array([5, 7, 9, 11, 13])
#grad_descent(x, y)

# new grad

# find derivative of x^2




