# -*- coding:utf-8 -*-

from keras.models import load_model
import numpy as np
import pandas as pd

BETS = {
    'b365': [3.10, 3.25, 2.38],
    'bw': [3.1, 3.3, 2.35],
    'iw': [3.1, 3.10, 2.3],
    'lb': [3, 3.25, 2.3],
    'wh': [2.9, 3.40, 2.38]
}

PREDICT_RS = {}


def predict(model_name, data):
    model = load_model(model_name)
    result = model.predict(data)
    return result


def analysis():
    around = 0.1
    for k, v in PREDICT_RS.iteritems():
        ds = pd.read_csv('datas/{}.csv'.format(k))
        ds.columns = ['H', 'D', 'A', 'R', 'P']
        ds = ds.loc[ds['P'] >= v-v*around].loc[ds['P'] <= v+v*around]
        H_count = D_count = P_count = 0
        if not ds.empty:
            H_count = ds.loc[ds['R'] == 1].shape[0]
            D_count = ds.loc[ds['R'] == 0].shape[0]
            P_count = ds.loc[ds['R'] == -1].shape[0]
        else:
            print 'empty'
        print H_count, D_count, P_count


if __name__ == '__main__':
    for k, v in BETS.iteritems():
        if v:
            data = np.array([
                [v, ],
            ])
            psr = predict('models/{}'.format(k), data)[0][0]
            print k, psr
            PREDICT_RS[k] = psr
    analysis()



