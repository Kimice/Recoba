# -*- coding:utf-8 -*-

import math
import datetime
import sqlite3
from sqlalchemy import create_engine
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


def init_conn():
    conn = sqlite3.connect('datas/database.sqlite')
    return conn


def gen_data(bet_company):
    conn = init_conn()
    cursor = conn.cursor()
    result = cursor.execute(
        'select A.home_team_goal, A.away_team_goal, '
        ' A.{hw}, A.{dd}, A.{aw} '
        # 'A.B365H, A.B365D, A.B365A, '
        # 'A.BWH, A.BWD, A.BWA, '
        # 'A.IWH, A.IWD, A.IWA, '
        # 'A.LBH, A.LBD, A.LBA, '
        # 'A.WHH, A.WHD, A.WHA, '
        # 'A.SJH, A.SJD, A.SJA, '
        # 'A.VCH, A.VCD, A.VCA, '
        # 'A.GBH, A.GBD, A.GBA, '
        # 'A.BSH, A.BSD, A.BSA '
        'from Match as A '
        'where A.{hw} is not NULL and A.{dd} is not NULL and A.{aw} is not NULL;'.format(
            hw=str(bet_company[0]).upper(),
            dd=str(bet_company[1]).upper(),
            aw=str(bet_company[2]).upper(),
        )
    ).fetchall()
    conn.close()
    return result


def train(data_set, rs_name):
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
        data_set[data_set.columns[2:-1]], data_set[data_set.columns[-1]], train_size=0.90)

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

    result = np.concatenate([X_test.reshape(X_test.shape[0], X_test.shape[2]),
                             Y_test, P_test], axis=1)
    np.savetxt('datas/{}.csv'.format(rs_name), result, delimiter=',')

    plt.figure(figsize=(10, 10))
    plt.plot(P_test[:, 0], label='pred')
    plt.plot(Y_test[:, 0], label='real')
    plt.legend(loc=0)
    plt.savefig('pics/test-compare.png', dpi=100)
    plt.show()
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
        self.activation_lstm = kwargs.get('activation_lstm', 'relu')
        self.activation_dense = kwargs.get('activation_dense', 'linear')
        self.activation_last = kwargs.get('activation_last', 'linear')
        self.dense_layer = kwargs.get('dense_layer', 2)
        self.lstm_layer = kwargs.get('lstm_layer', 2)
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', 20)
        self.batch_size = kwargs.get('batch_size', 100)
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.optimizer = kwargs.get('optimizer', 'adam')

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
    if s > 0:
        return 1
    elif s < 0:
        return -1
    else:
        return 0


BET_COMPANIES = [
    ('b365h', 'b365d', 'b365a'),
    ('bwh', 'bwd', 'bwa'),
    ('iwh', 'iwd', 'iwa'),
    ('lbh', 'lbd', 'lba'),
    ('whh', 'whd', 'wha'),
    # ('sjh', 'sjd', 'sja'),
    # ('vch', 'vcd', 'vca'),
    # ('gbh', 'gbd', 'gba'),
    # ('bsh', 'bsd', 'bsa'),
]


def del_data(bet_company):
    np.random.seed(1234)
    data = gen_data(bet_company)
    columns = ['hg', 'ag']
    columns.extend(bet_company)
    data_set = pd.DataFrame(data, columns=columns)
    data_set.dropna(how='any', inplace=True)
    data_set['rs'] = data_set.apply(
        lambda row: get_result(row['hg'] - row['ag']),
        axis=1
    )
    print data_set.info()
    bet_name = bet_company[0][:-1]
    model = train(data_set, bet_name)
    model.save('models/{}_match'.format(bet_name))
    # print model.predict(np.array([[[2.10, 3.20, 3.20], ]]))


if __name__ == '__main__':
    for bc in BET_COMPANIES:
        del_data(bc)
