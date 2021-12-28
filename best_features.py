import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression


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
    return calc/x.shape[0]


def J(th, x, y):
    sigma = np.sum(np.square(h(th, x) - y))
    return sigma/(2*x.shape[0])


def h(th, x):
    return x.dot(th)


def greedy(x, y):
    pass


def MI(x, y):
    global BEST_F_SIZE
    names = pd.DataFrame()
    for k in range(x.shape[1]):
        names[f'n{k}'] = x[:, k]

    mi_reg = mutual_info_regression(names, y, random_state=0)
    while len(mi_reg) > BEST_F_SIZE:
        features_to_drop = []
        for ind, prob in enumerate(mi_reg):
            if prob == 0 or prob == min(mi_reg):
                features_to_drop.append(ind)

        for feat in features_to_drop:
            names = names.drop(names.columns[feat], axis=1)

        mi_reg = mutual_info_regression(names, y, random_state=0)

    winners = []
    for k in range(x.shape[1]):
        if f'n{k}' in names:
            winners.append(f'n{k}')
    return winners


TEST_SIZE = 0.3  # Our size of TEST group when spliting data to train and test.
BEST_F_SIZE = 5
white = np.arange(0, 5)  # 0
grey = np.arange(5, 11)  # 1
black = np.arange(11, 17)  # 2


dig, labels = load_digits(return_X_y=True)
x_normal = StandardScaler().fit_transform(dig)


cpy = dig

# changing features values to: 0, 1, 2
for i in range(cpy.shape[0]):
    for j in range(cpy.shape[1]):
        if white.min() <= cpy[i][j] <= white.max():
            cpy[i][j] = 0
        elif grey.min() <= cpy[i][j] <= grey.max():
            cpy[i][j] = 1
        elif black.min() <= cpy[i][j] <= black.max():
            cpy[i][j] = 2

best_feat = MI(cpy, labels)
print(best_feat)
# x_train, x_test, y_train, y_test = train_test_split(x_normal, labels, test_size=TEST_SIZE, random_state=0)
#
# log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", penalty="l2", C=1)
# log_reg.fit(x_train, y_train)
#
# pred = log_reg.predict(x_test)
# prob_pred = log_reg.predict_proba(x_test)
# print(f'{pred}\n{prob_pred}')
