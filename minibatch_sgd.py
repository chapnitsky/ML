import numpy as np
import math
from matplotlib import style
from matplotlib import pyplot as plt
import time
import os

alpha = [0.1, 0.01, 0.001]


def timer(f, *args):
    """
    call f with args
    :param f:
    :param arg:
    :return:
    """
    start = time.time()
    f(*args)
    end = time.time()
    return end - start


def GradJ(th, x, y):
    """
    Calculates GRAD(J(theta))
    :return:
    Gradient J list
    """
    calc = x.T.dot(h(th, x) - y)
    return calc / x.shape[0]


def J(th, x, y):
    sigma = np.sum((h(th, x) - y) ** 2)
    return sigma / (2 * x.shape[0])


def h(th, x):
    return x.dot(th)


def GD(x, y):
    """
    Gradient Descent with 3 types of alpha: 0.1 0.01 0.001
    :return:
    theta
    """
    eps = 0.01
    K = 100
    colors = ['go', 'rs', 'b^']
    global alpha
    for index, alp in enumerate(alpha):
        if not alp:
            continue
        new_th = np.ones((x.shape[1], 1))
        prev_j = J(new_th, x, y)
        for k in range(1, K + 1):  # K is max iterations
            if k > 1 and abs(J(new_th, x, y) - prev_j) < eps:  # ||J(theta) -  previous || < epsilon
                break
            prev_j = J(new_th, x, y)
            new_th = new_th - alp * GradJ(new_th, x, y)
            if k == 1:
                plt.plot(k, J(new_th, x, y), colors[index], label=f'alpha: {alp}')
            else:
                plt.plot(k, J(new_th, x, y), colors[index])

    plt.title(f'Default Gradient Descent')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('J(theta)')
    style.use('ggplot')
    plt.show()


def stochastic(x, y):
    global alpha
    alp = alpha[0]  # alpha = 0.1 optimal
    eps = 0.001
    K = 100
    i = 0
    n = x.shape[1]
    m = x.shape[0]
    theta = np.ones((n, 1))
    prev_j = J(theta, x, y)
    prev_t = theta
    for k in range(1, K + 1):  # K is max iterations
        if k > 1 and abs(J(theta, x, y) - prev_j) < eps and abs(theta - prev_t) < eps:
            # ||J(theta) -  previous || < epsilon  & ||theta - previous|| < eps
            break
        prev_j = J(theta, x, y)
        prev_t = theta
        calc = alp * x[i % m] * (h(theta, x[i]) - y[i])
        calc = calc.reshape((n, 1))
        theta = theta - calc

        i += 1
        if k == 1:
            plt.plot(k, J(theta, x, y), 'k,', label='Stochastic')
        else:
            plt.plot(k, J(theta, x, y), 'k,')

    plt.title(f'Stochastic with optimal alpha: {alp}')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('J(theta)')
    style.use('ggplot')
    plt.show()


def mini_batch(x, y):
    alp = alpha[0]  # alpha = 0.1 optimal
    T = 100  # num of groups
    n = x.shape[1]
    m = x.shape[0]
    y_new = np.reshape(y, (m, 1))
    N = math.ceil(m / T)  # each group size
    theta = np.ones((n, 1))
    K = 100
    for k in range(0, K + 1):
        x_batch = x[k % m: (k + N) % m, :]
        y_batch = y_new[k % m: (k + N) % m, :]
        theta = theta - alp * GradJ(theta, x_batch, y_batch)
        if k == 0:
            plt.plot(k, J(theta, x, y), 'k,', label='Mini-Batch')
        else:
            plt.plot(k, J(theta, x, y), 'k,')
    plt.title(f'Mini-Batch with group size: {N}')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('J(theta)')
    style.use('ggplot')
    plt.show()



data = np.genfromtxt(f'{os.getcwd()}/cancer.csv', delimiter=',')
X = data
Y = X[:, -1]  # last column
X = np.delete(X, -1, 1)

sqrt_diff = np.std(X, axis=0)
avg = np.mean(X, axis=0)
tmp = X - avg
X = tmp / sqrt_diff
# print(f'{np.std(X)} : {np.mean(X)}')  # std = 1, mean = avg = 0
one = np.ones((X.shape[0], 1))
X = np.append(one, X, axis=1)  # adding ones to the matrix X
#
# print(timer(mini_batch, X, Y))
# print(timer(stochastic, X, Y))
print(timer(GD, X, Y))