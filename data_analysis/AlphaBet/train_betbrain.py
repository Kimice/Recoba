# -*- coding:utf-8 -*-

import math
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

ALL_COLS = [
    'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',
    'LBH', 'LBD', 'LBA', 'VCH', 'VCD', 'VCA', 'WHH', 'WHD', 'WHA',
    'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA',
    'BbOU', 'BbMx>2.5', 'BbMx<2.5', 'BbAv>2.5', 'BbAv<2.5',
    'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA'
]


def train(data_set):
    data_set = pd.DataFrame(data_set, columns=[
        'BbAvH', 'BbAvD', 'BbAvA',
        'BbAv>2.5', 'BbAv<2.5',
        'Hc', 'Dc', 'Ac',
        # 'FTR_H', 'FTR_D', 'FTR_A',
        'RS'
    ])
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
        data_set[data_set.columns[:-1]], data_set[data_set.columns[-1]], train_size=0.90)

    X_train_origin, X_test_origin = X_train, X_test
    X_train, X_test = X_train_origin[X_train_origin.columns[3:]], X_test_origin[X_test_origin.columns[3:]]

    X_train = np.reshape(np.array(X_train.values), (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(np.array(X_test.values), (X_test.shape[0], 1, X_test.shape[1]))
    Y_train = np.array(Y_train.values)
    Y_test = np.array(Y_test.values)

    if len(Y_train.shape) == 1:
        Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
        Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
    print X_train.shape, Y_train.shape
    print X_test.shape, Y_test.shape

    nn = NeuralNetwork()
    model = nn.NN_model(X_train, Y_train, X_test, Y_test)
    P_train = model.predict(X_train)
    P_test = model.predict(X_test)

    result = np.concatenate([X_test_origin, Y_test, P_test], axis=1)
    np.savetxt('datas/result.csv', result, delimiter=',')

    # plt.figure(figsize=(10, 10))
    # plt.plot(P_test[:, 0], label='pred')
    # plt.plot(Y_test[:, 0], label='real')
    # plt.legend(loc=0)
    # plt.savefig('pics/test-compare.png', dpi=100)
    # plt.show()
    # plt.figure(figsize=(10, 10))
    # plt.plot(P_train, label='pred')
    # plt.plot(Y_train, label='real')
    # plt.legend(loc=0)
    # plt.savefig('train-compare.png', dpi=100)
    # plt.show()

    print P_train.shape, P_test.shape, Y_train.shape, Y_test.shape, data_set.shape
    print 'Test Score: %.2f RMSE' % (math.sqrt(mean_squared_error(Y_test, P_test)))
    return model


class NeuralNetwork(object):
    def __init__(self, **kwargs):
        self.output_dim = kwargs.get('output_dim', 8)
        self.activation_lstm = kwargs.get('activation_lstm', 'sigmoid')
        self.activation_dense = kwargs.get('activation_dense', 'linear')
        self.activation_last = kwargs.get('activation_last', 'linear')
        self.dense_layer = kwargs.get('dense_layer', 2)
        self.lstm_layer = kwargs.get('lstm_layer', 2)
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', 20)
        self.batch_size = kwargs.get('batch_size', 500)
        self.loss = kwargs.get('loss', 'binary_crossentropy')
        self.optimizer = kwargs.get('optimizer', 'rmsprop')

    def NN_model(self, trainX, trainY, testX, testY):
        print "Training model is LSTM network!"
        input_dim = trainX[1].shape[1]
        output_dim = trainY.shape[1]
        model = Sequential()
        model.add(LSTM(output_dim=self.output_dim,
                       input_dim=input_dim,
                       activation=self.activation_lstm,
                       dropout_U=self.drop_out,
                       return_sequences=True
                       ))
        for i in range(self.lstm_layer-2):
            model.add(LSTM(output_dim=self.output_dim,
                           input_dim=self.output_dim,
                           activation=self.activation_lstm,
                           dropout_U=self.drop_out,
                           return_sequences=True
                           ))
        model.add(LSTM(output_dim=self.output_dim,
                       input_dim=self.output_dim,
                       activation=self.activation_lstm,
                       dropout_U=self.drop_out,
                       ))
        model.add(Dense(output_dim=output_dim,
                        input_dim=self.output_dim,
                        activation=self.activation_last))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        model.fit(x=trainX, y=trainY, nb_epoch=self.nb_epoch,
                  batch_size=self.batch_size, validation_data=(testX, testY))
        return model


def get_result(s):
    if s == 'H':
        return 1
    elif s == 'A':
        return -1
    elif s == 'D':
        return 0


USED_COLS = [
    'BbAv>2.5', 'BbAv<2.5', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'FTR'
]


def del_data():
    np.random.seed(1234)
    data_set = pd.DataFrame(pd.read_csv('datas/odds.csv'), columns=USED_COLS)
    data_set.dropna(how='any', inplace=True)
    data_set = data_set.loc[data_set['Bb1X2'] >= 10]\
        .loc[data_set['BbAvH'] >= 2.0].loc[data_set['BbAvA'] >= 2.0]
    # data_set.drop_duplicates()
    data_set['Hc'] = data_set.apply(
        lambda row: row['BbMxH']/row['BbAvH'],
        axis=1
    )
    data_set['Dc'] = data_set.apply(
        lambda row: row['BbMxD']/row['BbAvD'],
        axis=1
    )
    data_set['Ac'] = data_set.apply(
        lambda row: row['BbMxA']/row['BbAvA'],
        axis=1
    )
    data_set['RS'] = data_set.apply(
        lambda row: get_result(row['FTR']),
        axis=1
    )
    # data_set = pd.DataFrame(data_set, columns=[
    #     'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'Hc', 'Dc', 'Ac', 'RS'
    # ])
    # data_set = pd.get_dummies(data_set)
    print data_set.info(), data_set.shape
    model = train(data_set)
    # print model.predict(np.array([[[2.10, 3.20, 3.20], ]]))


if __name__ == '__main__':
    del_data()
