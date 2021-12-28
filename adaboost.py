import numpy as np
from sklearn.datasets import load_digits
from copy import deepcopy


def H(h_triplets: dict, x, y) -> bool:
    """
return True if belongs to class 1
return False if not belongs to class 1
"""
    global M
    global one
    global minus_one
    res = []
    for i in range(M):
        sum = 0
        for ind, trip in enumerate(h_triplets.values()):
            alpha = trip[2]
            sum += alpha * trip[5][i]
        res.append(sum > 0)

    good_guess = 0
    for ind, guess in enumerate(res):
        if guess and y[ind] == one:  # correct guess, it is class 1
            good_guess += 1
        elif not guess and y[ind] == minus_one:  # correct guess, it is class -1
            good_guess += 1

    score = good_guess / M
    return score


def adaboost10(h: dict, x, y):
    global M
    global N
    global one
    global minus_one
    eps_vec = np.zeros((N, 1))
    W = np.full((M, 1), 1 / M)
    T = 10
    best_h = {}
    wrongs = {}
    while len(best_h.keys()) != T:
        inp_guess = {}
        for ind, triplet in enumerate(h.values()):
            inp_guess[ind] = []
            wrong_ind = []
            bigger_than = (triplet[0] == '>')
            thres = triplet[1]
            for index, num in enumerate(x[:, ind]):  # on collumn ind, attr: n{ind}
                if (bigger_than and num > thres) or (not bigger_than and num < thres):  # predict that is class 1
                    inp_guess[ind].append(1)
                    if y[index] != one:
                        wrong_ind.append(index)  # Wrong input index
                else:  # predict that is class -1
                    inp_guess[ind].append(-1)
                    if y[index] != minus_one:
                        wrong_ind.append(index)

            tmp_eps = 0
            for i in wrong_ind:
                tmp_eps += W[i]

            eps_vec[ind] = tmp_eps
            wrongs[ind] = wrong_ind

        # updating h and alpha
        min_eps = min(eps_vec)  # minimal epsilon in time: t
        if min_eps == 0:
            print('100% success, pick 2 different digits')
            return
        min_h = np.argmin(eps_vec)  # best h in time: t
        cur_alpha = float(0.5 * np.log((1 - min_eps) / min_eps))
        updated_h = (h[min_h][0], h[min_h][1], cur_alpha, None, None, inp_guess[int(min_h)])
        h[min_h] = updated_h
        best_h[min_h] = True

        # weights
        min_wrongs = wrongs[int(min_h)]
        for i in range(len(W)):
            if i in min_wrongs:
                W[i] = 0.5*W[i]/min_eps
            else:
                W[i] = 0.5*W[i]/(1 - min_eps)

    return h, list(best_h.keys())


def find_bests(h: dict, x, y):
    global M
    global N
    global one
    global minus_one
    eps_vec = {}
    W = np.full((M, 1), 1 / M)
    best_h = {}

    for ni in range(N):
        eps_vec[ni] = []
        wrongs = {}
        inp_guess = {}
        for ind, triplet in enumerate(h.values()):
            inp_guess[ind] = []
            wrong_ind = []
            bigger_than = (triplet[0] == '>')
            thres = triplet[1]
            for index, num in enumerate(x[:, ni]):  # on collumn ni, attr: n{ni}
                if (bigger_than and num > thres) or (not bigger_than and num < thres):  # predict that is class 1
                    inp_guess[ind].append(1)
                    if y[index] != one:
                        wrong_ind.append(index)  # Wrong input index
                else:  # predict that is class -1
                    inp_guess[ind].append(-1)
                    if y[index] != minus_one:
                        wrong_ind.append(index)

            tmp_eps = 0
            for i in wrong_ind:
                tmp_eps += 1 / M
            eps_vec[ni].append(tmp_eps)
            wrongs[ind] = wrong_ind

        # updating h and alpha
        min_eps = min(eps_vec[ni])  # minimal epsilon in time: t
        if min_eps == 0:
            print('100% success, pick 2 different digits')
            return
        minimal = min(eps_vec[ni])
        min_h = eps_vec[ni].index(minimal)  # best h in time: t
        updated_h = (h[min_h][0], h[min_h][1], 1, minimal, wrongs[min_h], inp_guess[min_h])
        best_h[ni] = updated_h

    return h, best_h

PREDS = [0, 8]  # this combination is the most similar to each other
one = PREDS[0]
minus_one = PREDS[1]

dig, labels = load_digits(return_X_y=True)
dig = dig[np.logical_or(labels == one, labels == minus_one)]
labels = labels[np.logical_or(labels == one, labels == minus_one)]

X = deepcopy(dig)
M = X.shape[0]
N = X.shape[1]

np.random.seed()
oper = np.array(['>', '<'])
thresholds = np.array([i for i in range(0, 17)])
init_alpha = 1
h_funcs = []
for i in range(N):
    h_funcs.append((np.random.choice(oper), np.random.choice(thresholds), init_alpha))

h_dict = {key: value for key, value in zip(range(N), h_funcs)}
h_cpy = deepcopy(h_dict)



h_cpy, best = find_bests(h_cpy, X, labels)  # bests is a tuple of minimal hi info: (operator, threshold, epsilon, wrong indexes list, guesses)
print('Best hi for each attribute:')
for key, triplet in h_cpy.items():
    print(f'\tn{key}:\t({triplet[0]}, {triplet[1]})')

best_score = H(best, X, labels)
print(f'\nBest 64 hi score:\n\t{best_score}\n')


h_dict, top10 = adaboost10(h_dict, X, labels)
print(f'Best 10 hi :')
for ind in top10:
    print(f'\th{ind}:\t({h_dict[ind][0]}, {h_dict[ind][1]})\talpha:\t{h_dict[ind][2]}')

h_top = {count: h_dict[top_ind] for count, top_ind in enumerate(top10)}

ten_score = H(h_top, X, labels)
print(f'\nBest 10 hi score:\n\t{ten_score}')