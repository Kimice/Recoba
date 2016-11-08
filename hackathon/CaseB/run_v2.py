# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.feature_selection import (
    RFE,
    SelectFromModel
)
from sklearn.linear_model import LogisticRegression

from prepare import PrepareWorker
from predict import PredictWorker


def run():
    file_path = str(sys.argv[1]).strip().rstrip('/')
    c_data = PrepareWorker(file_path).prepare_click_train_data()
    p_data, log_ids = PrepareWorker(file_path).prepare_predict_data()
    c_data = pd.DataFrame(Imputer().fit_transform(c_data))
    p_data = pd.DataFrame(Imputer().fit_transform(p_data))
    x_click = np.array(c_data.iloc[:, 2:])
    y_click = np.array(c_data.iloc[:, 1])
    x_pred = np.array(p_data)

    # rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10).fit(x_click, y_click)
    # rfe = SelectFromModel(LogisticRegression())
    # rfe.fit(x_click, y_click)
    # x_click = rfe.transform(x_click)
    # x_pred = rfe.transform(x_pred)

    p_click = PredictWorker(x_click, y_click, x_pred).fit_and_predict()

    # b_data = PrepareWorker(file_path).prepare_buy_train_data()
    # x_buy = np.array(b_data.iloc[:, 2:])
    # y_buy = np.array(b_data.iloc[:, 0])
    # x_pred = np.insert(x_pred, 0, p_click, axis=1)
    # p_buy = PredictWorker(x_buy, y_buy, x_pred).fit_and_predict()

    # with open('/output/result.txt', 'w') as f:
    #     for i in xrange(len(log_ids)):
    #         # if int(p_click[i]) or int(p_buy[i]):
    #         f.write('\t'.join(
    #             (
    #                 str(log_ids[i]),
    #                 str(int(p_click[i])),
    #                 str(int(p_buy[i]))
    #             )
    #         ) + '\n')
    return 0


if __name__ == '__main__':
    run()
