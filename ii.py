import os
import time
import copy
import random
from tqdm import tqdm
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from aijack.attack import Poison_attack_sklearn

np.random.seed(77)
random.seed(77)

# Prepare Dataset
digits = load_digits()
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
# print(digits.data.shape)
# Create a classifier: a support vector classifier
clf = SVC(kernel="linear", gamma=0.001)
# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)
# Learn the digits on the train subset
clf.fit(X_train, y_train)
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
# Train the target classification model
#######clf = SVC(kernel="linear")
ac_score = metrics.accuracy_score(y_test, predicted)
cl_report = metrics.classification_report(y_test, predicted)
print(cl_report)

# Poisoning Attack
# We pick one training image and add poisoning noise to it.

# initial point
initial_idx = np.where(y_train == 7)[0][42]
xc = copy.deepcopy(X_train[initial_idx, :])
yc = y_train[initial_idx]

train_idx = random.sample(list(range(1, X_train.shape[0])), 100)
X_train_ = copy.copy(X_train[train_idx, :])
y_train_ = copy.copy(y_train[train_idx])
y_train_ = np.where(y_train_ == 7, 1, -1)
y_test_ = np.where(y_test == 7, 1, -1)

# You can execute poison attack with only one line.
attacker = Poison_attack_sklearn(clf, X_train_, y_train_, t=0.5)
xc_attacked, log = attacker.attack(xc, 1, X_test, y_test_, num_iterations=200)

# add poinsoned data
clf = SVC(kernel="linear", gamma=0.001)
clf.fit(X_train_, y_train_)
print("before attack: ", clf.score(X_test, y_test_))
clf = SVC(kernel="linear", gamma=0.001)
clf.fit(
    np.concatenate([X_train_, xc_attacked.reshape(1, -1)]),
    np.concatenate([y_train_, [-1]]),
)
print("after attack: ", clf.score(X_test, y_test_))
