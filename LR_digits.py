import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

PREDS = [1, 8]  # this combination is the most interesting one, in almmst every other combination the AUC is 1
TEST_SIZE = 0.3  # Our size of TEST group when spliting data to train and test. high value will cause: a bigger ROC curve and a lower AUC
one = PREDS[0]  # our class we want to classify
zero = PREDS[1]

dig, labels = load_digits(return_X_y=True)
dig = dig[np.logical_or(labels == one, labels == zero)]
labels = labels[np.logical_or(labels == one, labels == zero)]

x_normal = StandardScaler().fit_transform(dig)

x_train, x_test, y_train, y_test = train_test_split(x_normal, labels, test_size=TEST_SIZE, random_state=0)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

pred = log_reg.predict(x_test)
acc = log_reg.score(x_test, y_test)
confusion = pd.crosstab(y_test, pred, rownames=['Real'], colnames=['Predictions'])
mfpr = ((confusion[one][zero]) / (confusion[one][zero] + confusion[zero][zero]))
mtpr = ((confusion[one][one]) / (confusion[one][one] + confusion[zero][one]))
precision = ((confusion[one][one]) / (confusion[one][one] + confusion[one][zero]))

print(f'Accuracy: {acc}\nPrecision: {precision}\nfpr: {mfpr}\ntpr: {mtpr}')
sn.heatmap(confusion, annot=True)
plt.show()

prob_pred = log_reg.predict_proba(x_test)[::, 1]

ones_zeros = y_test
for index, number in enumerate(y_test):
    if number == one:
        ones_zeros[index] = 0
    else:
        ones_zeros[index] = 1

fpr, tpr, threshold = metrics.roc_curve(ones_zeros, prob_pred)
auc = metrics.roc_auc_score(ones_zeros, prob_pred)
plt.title(f'ROC curve with AUC: {round(auc, 4)}')
plt.xlim(-0.01, 1.01)
plt.xlabel('FPR')
plt.ylim(-0.01, 1.01)
plt.ylabel('TPR')
plt.plot(fpr, tpr, "g-", label="ROC Curve")
plt.plot(np.arange(0, 2, 1), np.arange(0, 2, 1), 'r--', label='y = x')
plt.plot(0, 0, 'ko', label='S = 0')
plt.plot(1, 1, 'bo', label='S = 1')
plt.legend()
plt.show()
