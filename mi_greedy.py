import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from copy import deepcopy


TEST_SIZE = 0.15  # Our size of TEST group when spliting data to train and test.
BEST_F_SIZE = 5  # NUMBER OF BEST FEATURES WE WANT TO FIND
white = np.arange(0, 5)  # 0
grey = np.arange(5, 11)  # 1
black = np.arange(11, 17)  # 2
val_types = [0, 1, 2]

dig, labels = load_digits(return_X_y=True)


def findOneBest(bests, x, y):
    global TEST_SIZE

    cur = pd.DataFrame()
    for feat in bests:  # building current features data frame
        cur[feat] = x[feat]

    max = 0
    best_name = -1
    for col_name in x.columns:
        if col_name in bests:
            continue
        cur[col_name] = x[col_name]  # adding new feature to check
        x_normal = StandardScaler().fit_transform(cur)

        x_train, x_test, y_train, y_test = train_test_split(x_normal, y, test_size=TEST_SIZE, random_state=0)
        log_reg = LogisticRegression(multi_class="multinomial", penalty="l2",
                                     C=1)  # C=1 is optimal for this data, gives 96.6667%
        log_reg.fit(x_train, y_train)
        acc = log_reg.score(x_test, y_test)
        if acc > max:
            max = acc
            best_name = col_name

        cur = cur.drop(col_name, axis=1)
    return best_name, max


def greedy(x, y):
    global TEST_SIZE
    global BEST_F_SIZE
    best_yet = []
    scores = []
    df = pd.DataFrame()
    for index in range(x.shape[1]):
        df[f'n{index}'] = x[:, index]

    while len(best_yet) != BEST_F_SIZE:
        best_col, col_score = findOneBest(best_yet, df, y)
        best_yet.append(best_col)
        scores.append(col_score)

    return best_yet, scores


tmp = deepcopy(dig)
start = time.time()
greedy_best, greedy_score = greedy(tmp, labels)
end = time.time()
greed_time = end - start

print(f'Greedy group of {BEST_F_SIZE}: \n\t{greedy_best}')
print(
    f'Final Score: \n\t{greedy_score[-1]}\nScore in iterations: \n\t{greedy_score}\nRuntime: \n\t{float(greed_time)} seconds\n\n')
axis_x = [i for i in range(1, BEST_F_SIZE + 1)]
plt.plot(axis_x, greedy_score, 'r-', label='Greedy, 5 features')
plt.legend(loc='best')
plt.ylim(bottom=0, top=1)
plt.xlabel('Best Features Size')
plt.ylabel('Score')
plt.style.use('fivethirtyeight')
plt.show()


def history(x, y):  # Gather fast information from x Matrix, for probability
    y_dict = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
    x_dict = {}
    x_dict_len = {}
    zeros = 0
    ones = 0
    twos = 0
    for col in x.columns:
        x_dict[col] = {'0': [], '1': [], '2': []}
        for row in range(x.shape[0]):
            x_dict[col][f'{int(x[col][row])}'].append(row)
        zeros += len(x_dict[col]['0'])
        ones += len(x_dict[col]['1'])
        twos += len(x_dict[col]['2'])
        x_dict_len[col] = {'0': len(x_dict[col]['0']), '1': len(x_dict[col]['1']), '2': len(x_dict[col]['2'])}

    x_dict_total = {'0': zeros, '1': ones, '2': twos}

    for ind, yi in enumerate(y):
        yi = int(yi)
        y_dict[f'{yi}'].append(ind)
    y_len = {'0': len(y_dict['0']), '1': len(y_dict['1']), '2': len(y_dict['2']), '3': len(y_dict['3']),
             '4': len(y_dict['4']), '5': len(y_dict['5']), '6': len(y_dict['6']), '7': len(y_dict['7']),
             '8': len(y_dict['8']), '9': len(y_dict['9'])}

    return x_dict, x_dict_len, y_dict, y_len, x_dict_total


def P(jj, kk, x, y, xi, xx: tuple, yy: tuple, xtotal):  # calculates probability
    global val_types
    if jj is None and kk is not None:
        return float(yy[1][f'{kk}'] / len(y))  # p_yi_k

    if kk is None and jj is not None:
        return float(xtotal[f'{jj}'] / (x.shape[0] * x.shape[1]))  # p_xi_j

    list_x = xx[0][xi][f'{jj}']
    list_y = yy[0][f'{kk}']
    common_occurences = len(set(list_x).intersection(list_y))
    p_xi_j_given_yk = float(common_occurences / len(y))
    # print(f'P(xi={jj} | yi={kk})  [{xi}] = {p_xi_j_given_yk}')
    return p_xi_j_given_yk


def P_MI(x, y, xdict, xlen, ydict, ylen, xtotal):  # Using for MI formula
    global val_types
    mi_list = []
    history_k = [-1] * 10  # classes
    history_j = [-1] * 3  # x vals
    for xi in x.columns:
        sum = 0
        for j in val_types:  # j is 0 or 1 or 2
            for k in np.arange(0, 10):  # y is from 0 to 9
                a = P(j, k, x, y, xi, (xdict, xlen), (ydict, ylen), xtotal)
                if a == float(0):
                    continue
                if history_j[j] != -1:
                    b = history_j[j]
                else:
                    b = P(j, None, x, y, None, (xdict, xlen), (ydict, ylen), xtotal)
                    history_j[j] = b

                if history_k[k] != -1:
                    c = history_k[k]
                else:
                    c = P(None, k, x, y, None, (xdict, xlen), (ydict, ylen), xtotal)
                    history_k[k] = c

                sum += a * np.log2(a / (b * c))
        mi_list.append(sum)
    return mi_list


def MI(x, y):
    global BEST_F_SIZE
    global TEST_SIZE
    names = pd.DataFrame()
    for k in range(x.shape[1]):
        names[f'n{k}'] = x[:, k]

    mi_reg = [0] * (BEST_F_SIZE + 1)
    while len(mi_reg) > BEST_F_SIZE:
        xdict, xlen, ydict, ylen, xtotal = history(names, y)
        mi_reg = P_MI(names, y, xdict, xlen, ydict, ylen, xtotal)

        if len(mi_reg) == BEST_F_SIZE:  # FOUND, just need their final score by L.R
            x_normal = StandardScaler().fit_transform(names)
            x_train, x_test, y_train, y_test = train_test_split(x_normal, y, test_size=TEST_SIZE, random_state=0)
            log_reg = LogisticRegression(multi_class="multinomial", penalty="l2",
                                         C=1)  # C=1 is optimal for this data, gives 96.6667%
            log_reg.fit(x_train, y_train)
            mi_scor = log_reg.score(x_test, y_test)
            return list(names.columns), mi_scor

        worst_feat = mi_reg.index(min(mi_reg))
        cur_name = names.columns[int(worst_feat)]
        names = names.drop(labels=cur_name, axis=1)


cpy = deepcopy(dig)

# changing features values to: 0, 1, 2   for MI
for i in range(cpy.shape[0]):
    for j in range(cpy.shape[1]):
        if white.min() <= cpy[i][j] <= white.max():
            cpy[i][j] = 0
        elif grey.min() <= cpy[i][j] <= grey.max():
            cpy[i][j] = 1
        elif black.min() <= cpy[i][j] <= black.max():
            cpy[i][j] = 2

before = time.time()
mi, mi_score = MI(cpy, labels)
mi_time = time.time() - before
print(
    f'\nMutual Information group of {BEST_F_SIZE}: \n\t{mi}\nFinal Score: \n\t{mi_score}\nRuntime: \n\t{mi_time} seconds\n\n\n')

scores = {}
times = {}

cpy = deepcopy(dig)
for i in range(cpy.shape[0]):  # changing features values to: 0, 1, 2   for MI
    for j in range(cpy.shape[1]):
        if white.min() <= cpy[i][j] <= white.max():
            cpy[i][j] = 0
        elif grey.min() <= cpy[i][j] <= grey.max():
            cpy[i][j] = 1
        elif black.min() <= cpy[i][j] <= black.max():
            cpy[i][j] = 2

# RANG = dig.shape[1], Takes too much time to calculate
RANG = 8
bottom = 5

for feat_size in range(bottom, RANG):
    BEST_F_SIZE = feat_size

    tmp = deepcopy(dig)
    start = time.time()
    greedy_best, greedy_score = greedy(tmp, labels)
    end = time.time()
    greed_time = end - start

    # print(f'Greedy group of {BEST_F_SIZE}: \n\t{greedy_best}')
    # print(f'Final Score: \n\t{greedy_score[-1]}\nRuntime: \n\t{float(greed_time)} seconds')

    for_cpy = deepcopy(cpy)
    mi, mi_score = MI(for_cpy, labels)
    mi_time = time.time() - end
    scores[feat_size] = (greedy_score[-1], mi_score)
    times[feat_size] = (greed_time, mi_time)
    # print(f'\nMutual Information group of {BEST_F_SIZE}: \n\t{mi}\nFinal Score: \n\t{mi_score}\nRuntime: \n\t{mi_time} seconds\n\n\n')

greed_scr = [tup[0] for tup in scores.values()]
mi_scr = [tup[1] for tup in scores.values()]
greed_t = [tup[0] for tup in times.values()]
mi_t = [tup[1] for tup in times.values()]
x_axis = [numm for numm in range(bottom, RANG, 1)]

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(2, figsize=(10, 12))

ax[0].plot(x_axis, greed_scr, 'r-', label='Greedy')
ax[0].plot(x_axis, mi_scr, 'g-', label='Mutual Information')
ax[0].set(xlabel='Best Features Size', ylabel='Score')
ax[0].get_xaxis().set_major_locator(MaxNLocator(integer=True))

ax[1].plot(x_axis, greed_t, 'r-', label='Greedy')
ax[1].plot(x_axis, mi_t, 'g-', label='Mutual Information')
ax[1].set(xlabel='Best Features Size', ylabel='Time (seconds)')
ax[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

plt.legend(loc='best')
ax[0].set_ylim(bottom=0, top=1)
ax[1].set_ylim(bottom=0, top=None)
plt.tight_layout()
plt.show()