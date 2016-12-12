# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.datasets import load_boston
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    LassoCV
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.svm import (
    SVC,
    OneClassSVM
)
from sklearn.preprocessing import Imputer
import seaborn as sns


data_set = load_boston()
X, Y = data_set.data.astype('|F16'), data_set.target.astype('|F16')
target = data_set.target
feature_names = data_set.feature_names
data_set = pd.DataFrame(data_set.data, columns=feature_names)

whole_df = pd.concat([data_set, pd.DataFrame(target, columns=['MEDV'])], axis=1)
print whole_df.describe()

# print data_set.shape
# print data_set.info()
# print data_set.describe()

# plt.hist(Y, bins=20)
# plt.suptitle('Boston Housing Prices in $1000s', fontsize=15)
# plt.xlabel('Prices in $1000s')
# plt.ylabel('Count')
# plt.savefig('MEDV.png', dpi=100)
# plt.show()

# # box-and-whisker univariate plots (plots of each individual variable)
# data_set.plot(kind='box', subplots=True, layout=(1, 13), sharex=True, sharey=False)
# plt.savefig('box-and-whisker-plots.png', dpi=100)
# plt.show()
#
# # histogram univariate plots
# data_set.hist()
# plt.savefig('histograms.png', dpi=100)
# plt.show()
#
# # scatter plot matrix multivariate plot
# scatter_matrix(data_set)
# plt.savefig('scatter-matrix.png', dpi=100)
# plt.show()

# sns.heatmap(whole_df)
sns.jointplot(whole_df['MEDV'], whole_df['LSTAT'], kind='kde')
plt.show()

validation_size = 0.20
seed = 7
# split dataset into training set (80%) and validation set (20%)
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(
    X, Y, test_size=validation_size, random_state=seed)

# 10-fold cross validation to estimate accuracy (split data into 10 parts; use 9 parts to train and 1 for test)
num_folds = 10
num_instances = len(X_train)
seed = 7
# use the 'accuracy' metric to evaluate models (correct / total)
scoring = 'neg_mean_squared_error'

# algorithms / models
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsRegressor()))
# models.append(('CART', DecisionTreeRegressor()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# models.append(('RFS', RandomForestRegressor()))
# models.append(('OC', OneClassSVM()))
# models.append(('GB', GradientBoostingRegressor()))
# models.append(('LNR', LinearRegression()))
# models.append(('LS', LassoCV()))

# evaluate each algorithm / model
# results = []
# names = []
# print("Scores for each algorithm:")
# for name, model in models:
#     kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
#     cv_result = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_result)
#     names.append(name)
#     print name, ":", cv_result.mean()

# # plot algorithm comparison (boxplot)
# fig = plt.figure()
# fig.suptitle('Algorithm comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.savefig('Algorithm-comparison.png')
# plt.show()

# using KNN to make predictions about the validation set
# knn = KNeighborsRegressor()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

