import pandas
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix
)
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from math import ceil
import numpy as np

from keras.models import Sequential
from keras.layers import (
    Dense,
    Activation,
    Merge,
    LSTM
)
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor


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
            pass
    return result, encoders


def base_model():
    model = Sequential()
    model.add(Dense(output_dim=1, input_dim=64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def analysis():
    url = "autos.data"
    features = ['symbol', 'losses', 'make', 'ftype', 'aspiration', 'doors', 'style', 'wheels',
                'eloc', 'wbase', 'length', 'width', 'height',  'weight', 'etype',
                'cylinders', 'esize', 'fsys', 'bore', 'stroke', 'compression',
                'power', 'rpm', 'cmpg', 'hmpg', 'price']
    dataset = pandas.read_csv(url, names=features)
    # del dataset['education']
    dataset = dataset.replace('?', np.NaN)
    dataset.dropna(how='any', inplace=True)
    # print dataset.info()
    # print(dataset.shape)
    # print(dataset.head(5))
    # print(dataset.describe())

    # Calculate the correlation and plot it
    # sns.heatmap(encoded_data.corr(), square=True)
    # plt.show()
    binary_data = pandas.get_dummies(dataset[['make', 'ftype', 'aspiration', 'doors', 'style',
                                              'wheels', 'eloc', 'etype', 'fsys', 'cylinders']])
    final_data = pandas.concat([binary_data,
                                dataset[['symbol', 'losses', 'wbase', 'length', 'width',
                                         'height',  'weight', 'esize', 'bore', 'stroke',
                                         'compression', 'power', 'rpm', 'cmpg', 'hmpg', 'price']]],
                               axis=1)
    # print final_data.info()
    # plt.subplots(figsize=(20, 30))
    # sns.heatmap(final_data.corr(), square=True)
    # plt.show()

    # show_each_columns_features(dataset)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        final_data[final_data.columns[:-1]], final_data["price"], train_size=0.70)
    scaler = StandardScaler()
    X_train = np.asarray(pandas.DataFrame(X_train.astype("float64"), columns=X_train.columns))
    X_test = np.asarray(X_test.astype("float64"))
    y_train = np.asarray(y_train.astype("float64"))
    y_test = np.asarray(y_test.astype("float64"))

    clf = KerasRegressor(build_fn=base_model, nb_epoch=1000, batch_size=5, verbose=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # for k, v in enumerate(res):
    #     print '%s : %s' % (int(v), int(y_test[k]))

    plt.figure(figsize=(15, 15))
    plt.plot(y_pred, label='pred')
    plt.plot(y_test, label='real')
    plt.legend(loc=0)
    plt.savefig('compare.png', dpi=100)
    plt.show()

    # print clf.score(X_test, y_test)
    # print mean_squared_error(y_test, res)
    # print mean_absolute_error(y_test, res)


if __name__ == "__main__":
    analysis()
