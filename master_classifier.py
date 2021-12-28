import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import csv
from sklearn.metrics import plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import svm


def reflect_configs():
    global MODELS, x_train, x_test, y_train, y_test, LR_ACC_C, V1_ACC_C, VA_ACC_C, TOP_C

    li1 = [round(0.001, 3), round(0.01, 2), round(0.1, 1)]
    li2 = list(range(1, 31, 1))
    li1.extend(li2)

    li3 = list(range(1, 25, 1))

    for m, model_str in MODELS.items():

        cc = 0
        accuracy = 0
        best_m = 0
        num_li = li1
        if m == 'Ada':
            num_li = li3

        for num in num_li:
            tmp_str = model_str

            if m == 'Softmax':
                tmp_str += f'{num})'
            elif m == 'Ada':
                tmp_str += f'{num})'
            else:
                tmp_str += f'{num}))'

            model = eval(tmp_str)
            model.fit(x_train, y_train)
            acc = model.score(x_test, y_test)
            if m == 'Softmax':
                LR_ACC_C[num] = acc
            elif m == '1v1':
                V1_ACC_C[num] = acc
            elif m == 'Ada':
                ADA_ACC_EST[num] = acc
            else:
                VA_ACC_C[num] = acc

            if acc > accuracy:
                cc = num
                accuracy = acc
                best_m = tmp_str

        MODELS[m] = best_m
        TOP_C[m] = (cc, accuracy)


def reflect_feats():
    global MODELS, TOP_FEAT, GREEDY, Mi, LASSO, FEAT_SELECT, BEST_F_SIZE, N, cpy4, TEST_SIZE, Y, TYPES
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))
    fig.suptitle('Models / Filters')
    filt_counter = 0
    for filt, filt_str in FEAT_SELECT.items():
        tmp_filt = filt_str
        for i in range(2, N + 1, 1):
            BEST_F_SIZE = i

            if filt == 'lasso':
                col_inds = classo()
            else:
                col_names, group_acc = eval(tmp_filt)
                col_inds = [int(name[1:]) for name in col_names]

            filtered_normal = cpy4[:, col_inds]
            x_train4, x_test4, y_train4, y_test4 = train_test_split(filtered_normal, Y, test_size=TEST_SIZE,
                                                                    random_state=np.random.randint(0, 300 + 1))
            counter = 0
            for m, model_str in MODELS.items():
                tmp_str = model_str
                model = eval(tmp_str)
                model.fit(x_train4, y_train4)
                acc = model.score(x_test4, y_test4)
                if i == 5:
                    plot_confusion_matrix(model, x_test4, y_test4, ax=axes[counter][filt_counter],
                                          display_labels=list(TYPES.values()))
                    axes[counter][filt_counter].title.set_text(f'\n{m} by {filt}, acc ={acc}')

                if filt == 'lasso':
                    LASSO[m][i] = acc
                elif filt == 'Greedy':
                    GREEDY[m][i] = acc
                else:
                    Mi[m][i] = acc

                counter += 1

        filt_counter += 1

    plt.tight_layout(pad=4.0)
    plt.show()


def classo() -> list:
    global Y, BEST_F_SIZE, cpy1
    lass = Lasso(alpha=0.1, random_state=np.random.randint(0, 300 + 1), copy_X=True)
    lass.fit(cpy1, Y)
    best_feats = []
    for i, coef in enumerate(lass.coef_):
        if len(best_feats) == BEST_F_SIZE:
            return best_feats

        elif coef != 0:
            best_feats.append(i)

    return best_feats


def findOneBest(bests, df):
    global TEST_SIZE, Y

    cur = pd.DataFrame()
    for feat in bests:  # building current features data frame
        cur[feat] = df[feat]

    max = 0
    best_name = -1
    for col_name in df.columns:
        if col_name in bests:
            continue
        cur[col_name] = df[col_name]  # adding new feature to check
        x_norm = StandardScaler().fit_transform(cur)
        x_train2, x_test2, y_train2, y_test2 = train_test_split(x_norm, Y, test_size=TEST_SIZE,
                                                                random_state=np.random.randint(0, 300 + 1))

        log_reg = eval(MODELS['Softmax'])
        log_reg.fit(x_train2, y_train2)
        acc = log_reg.score(x_test2, y_test2)
        if acc > max:
            max = acc
            best_name = col_name

        cur = cur.drop(col_name, axis=1)
    return best_name, max


def MI() -> (list, float):
    global BEST_F_SIZE, TEST_SIZE, Y, MODELS, cpy2
    x = cpy2
    names = pd.DataFrame()
    for k in range(x.shape[1]):
        names[f'n{k}'] = x[:, k]

    mi_reg = [0] * (BEST_F_SIZE + 1)
    while len(mi_reg) > BEST_F_SIZE:
        mi_reg = mutual_info_classif(names, Y)

        if len(mi_reg) == BEST_F_SIZE:
            x_norma = StandardScaler().fit_transform(names)
            x_train3, x_test3, y_train3, y_test3 = train_test_split(x_norma, Y, test_size=TEST_SIZE, random_state=0)
            log_reg = eval(MODELS['Softmax'])
            log_reg.fit(x_train3, y_train3)
            mi_scor = log_reg.score(x_test3, y_test3)
            return list(names.columns), mi_scor

        worst_feat = np.argmin(mi_reg)

        cur_name = names.columns[int(worst_feat)]
        names = names.drop(labels=cur_name, axis=1)


def greedy() -> (list, float):
    global TEST_SIZE, BEST_F_SIZE, cpy3
    x = cpy3
    best_yet = []
    scores = []
    df = pd.DataFrame()
    for index in range(x.shape[1]):
        df[f'n{index}'] = x[:, index]

    while len(best_yet) != BEST_F_SIZE:
        best_col, col_score = findOneBest(best_yet, df)
        best_yet.append(best_col)
        scores.append(col_score)
    return best_yet, scores[-1]


def draw_configs():
    global LR_ACC_C, V1_ACC_C, VA_ACC_C, ADA_ACC_EST, TOP_C

    fig2, axes2 = plt.subplots(4, figsize=(20, 30))
    axes2[0].plot(list(LR_ACC_C.keys()), [num * 100 for num in LR_ACC_C.values()])
    axes2[0].set_title(f'Softmax C configuration graph, best C = {TOP_C["Softmax"][0]}')
    axes2[0].set_xlabel('C')
    axes2[0].set_ylabel('Accuracy')

    axes2[1].plot(list(V1_ACC_C.keys()), [num * 100 for num in V1_ACC_C.values()])
    axes2[1].set_title(f'1v1 C configuration graph, best C = {TOP_C["1v1"][0]}')
    axes2[1].set_xlabel('C')
    axes2[1].set_ylabel('Accuracy')

    axes2[2].plot(list(VA_ACC_C.keys()), [num * 100 for num in VA_ACC_C.values()])
    axes2[2].set_title(f'1vAll C configuration graph, best C = {TOP_C["1vA"][0]}')
    axes2[2].set_xlabel('C')
    axes2[2].set_ylabel('Accuracy')

    axes2[3].plot(list(ADA_ACC_EST.keys()), [num * 100 for num in ADA_ACC_EST.values()])
    axes2[3].set_title(f'Adaboost Estimators configuration graph, best estimators = {TOP_C["Ada"][0]}')
    axes2[3].set_xlabel('Estimators count')
    axes2[3].set_ylabel('Accuracy')

    plt.tight_layout(pad=3.0)
    plt.show()


def draw_feats():
    global GREEDY, Mi, LASSO, TOP_FEAT

    fig, axes3 = plt.subplots(nrows=4, ncols=3, figsize=(30, 20))

    count = 0
    for m_name, m_dict in GREEDY.items():
        search_string = 'features'
        feat_list = [int(num) for num in GREEDY[m_name].keys()]
        feat_acc = [num * 100 for num in GREEDY[m_name].values()]

        top_acc = max(feat_acc)
        top_num_feat = feat_list[feat_acc.index(top_acc)]
        if m_name == 'Ada':
            search_string = 'estimators'
        axes3[count][0].plot(feat_list, feat_acc, 'r-', label=f'{m_name} by Greedy')
        axes3[count][0].set_title(f'Best {search_string} number = {top_num_feat}, Accuracy = {top_acc}')
        axes3[count][0].set_xlabel(f'Number of {search_string}')
        axes3[count][0].set_ylabel('Accuracy')
        axes3[count][0].legend(loc='best')
        count += 1

    count = 0
    for m_name, m_dict in Mi.items():
        search_string = 'features'
        feat_list = [int(num) for num in Mi[m_name].keys()]
        feat_acc = [num * 100 for num in Mi[m_name].values()]

        top_acc = max(feat_acc)
        top_num_feat = feat_list[feat_acc.index(top_acc)]
        if m_name == 'Ada':
            search_string = 'estimators'
        axes3[count][1].plot(feat_list, feat_acc, 'g-', label=f'{m_name} by MI')
        axes3[count][1].set_title(f'Best {search_string} number = {top_num_feat}, Accuracy = {top_acc}')
        axes3[count][1].set_xlabel(f'Number of {search_string}')
        axes3[count][1].set_ylabel('Accuracy')
        axes3[count][1].legend(loc='best')
        count += 1

    count = 0
    for m_name, m_dict in LASSO.items():
        search_string = 'features'
        feat_list = [int(num) for num in LASSO[m_name].keys()]
        feat_acc = [num * 100 for num in LASSO[m_name].values()]

        top_acc = max(feat_acc)
        top_num_feat = feat_list[feat_acc.index(top_acc)]
        if m_name == 'Ada':
            search_string = 'estimators'
        axes3[count][2].plot(feat_list, feat_acc, 'b-', label=f'{m_name} by Lasso')
        axes3[count][2].set_title(f'Best {search_string} number = {top_num_feat}, Accuracy = {top_acc}')
        axes3[count][2].set_xlabel(f'Number of {search_string}')
        axes3[count][2].set_ylabel('Accuracy')
        axes3[count][2].legend(loc='best')

        count += 1

    plt.tight_layout(pad=3.0)
    plt.show()



path = ''
# We have 4 different models to check:
MODELS = {'Softmax': "LogisticRegression(penalty='l2', max_iter=300, multi_class='multinomial', solver='lbfgs', C=",
          '1v1': "OneVsOneClassifier(svm.LinearSVC(dual = False, C=",
          '1vA': "OneVsRestClassifier(svm.LinearSVC(dual = False, C=",
          'Ada': "AdaBoostClassifier(n_estimators="}
keys = list(MODELS.keys())


TEST_SIZE = 0.25
TYPES = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
FEAT_SELECT = {'lasso': 'classo()', 'MI': 'MI()', 'Greedy': 'greedy()'}

LR_ACC_C = {}
V1_ACC_C = {}
VA_ACC_C = {}
ADA_ACC_EST = {}
TOP_C = {}

TOP_FEAT = {}
GREEDY = {key: {} for key in keys}
Mi = {key: {} for key in keys}
LASSO = {key: {} for key in keys}
BEST_F_SIZE = 5
EPS = 0.01  # epsilon

data = np.loadtxt(open(path + "fetal_health.csv", "rb"), delimiter=",", skiprows=1)
X = data[:, :-1]
Y = data[:, -1]
M = X.shape[0]
N = X.shape[1]

x_normal = StandardScaler().fit_transform(X)
cpy1 = deepcopy(x_normal)
cpy2 = deepcopy(x_normal)
cpy3 = deepcopy(x_normal)
cpy4 = deepcopy(x_normal)

np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x_normal, Y, test_size=TEST_SIZE,
                                                    random_state=np.random.randint(0, 300 + 1))
for_cols = open(path + 'fetal_health.csv', 'r+')
cols = next(csv.reader(for_cols, delimiter=','))  # first line is the names of the columns

reflect_configs()
reflect_feats()
print(cols)
print(f'\n{LR_ACC_C}, \n{V1_ACC_C}, \n{VA_ACC_C}, \n{ADA_ACC_EST},\n{TOP_C}')
print(f'\n\n\n{GREEDY}\n\n{Mi}\n\n{LASSO}\n\n')
draw_configs()
draw_feats()
print(cols)
print(f'\n{LR_ACC_C}, \n{V1_ACC_C}, \n{VA_ACC_C}, \n{ADA_ACC_EST},\n{TOP_C}')
print(f'\n\n\n{GREEDY}\n\n{Mi}\n\n{LASSO}\n\n')
