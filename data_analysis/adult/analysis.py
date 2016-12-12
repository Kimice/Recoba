import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
import statsmodels.api as sm
import seaborn as sns
from math import ceil
import numpy as np


def show_each_columns_features(df):
    fig = plt.figure(figsize=(10, 10))
    cols = 5
    rows = ceil(float(df.shape[1]) / cols)
    for i, column in enumerate(df.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if df.dtypes[column] == np.object:
            df[column].value_counts().plot(kind="bar")
        else:
            df[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    plt.show()


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


def analysis():
    url = "adult.data"
    features = ['age', 'work_class', 'fnlwgt', 'education', 'education_num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'salary']
    dataset = pandas.read_csv(url, names=features)
    del dataset['education']
    dataset.dropna(how='any', inplace=True)
    # print dataset.info()
    # print(dataset.shape)
    # print(dataset.head(5))
    # print(dataset.describe())

    # Calculate the correlation and plot it
    encoded_data, encoders = number_encode_features(dataset)
    # sns.heatmap(encoded_data.corr(), square=True)
    # plt.show()

    binary_data = pandas.get_dummies(dataset)
    binary_data["salary"] = binary_data["salary_ >50K"]
    del binary_data["salary_ <=50K"]
    del binary_data["salary_ >50K"]
    # plt.subplots(figsize=(20, 30))
    # sns.heatmap(binary_data.corr(), square=True)
    # plt.show()

    # show_each_columns_features(dataset)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        binary_data[binary_data.columns[:-1]], binary_data["salary"], train_size=0.70)
    scaler = StandardScaler()
    X_train = pandas.DataFrame(scaler.fit_transform(X_train.astype("f64")), columns=X_train.columns)
    X_test = scaler.transform(X_test.astype("f64"))

    # cls = LogisticRegression()
    # cls.fit(X_train, y_train)
    # y_pred = cls.predict(X_test)
    # print "F1 score: %f" % f1_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(12, 12))
    # plt.subplot(2, 1, 1)
    # sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["salary"].classes_,
    #             yticklabels=encoders["salary"].classes_)
    # plt.ylabel("Real value")
    # plt.xlabel("Predicted value")
    # coefs = pandas.Series(cls.coef_[0], index=X_train.columns)
    # coefs.sort()
    # plt.subplot(2, 1, 2)
    # coefs.plot(kind="bar")
    # plt.show()

    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    print("Scores for each algorithm:")
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print "%s F1 score: %f" % (name, f1_score(y_test, y_pred))
    #
    # fig = plt.figure()
    # fig.suptitle('Algorithm comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # plt.savefig('Algorithm-comparison.png')
    # plt.show()
    #
    # knn = KNeighborsClassifier()
    # knn.fit(X_train, Y_train)
    # predictions = knn.predict(X_validation)
    # print(accuracy_score(Y_validation, predictions))
    # print(confusion_matrix(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))


if __name__ == "__main__":
    analysis()
