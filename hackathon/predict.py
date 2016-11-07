# -*- coding: utf-8 -*-

import time

from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class PredictWorker(object):
    def __init__(self, x, y, p_x=None, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test\
            = cross_validation.train_test_split(x, y, test_size=test_size)
        self.p_x = p_x
        self.scaler = StandardScaler()
        n_iter = max(10**6 / len(self.x_train), 1)
        self.classifier = SGDClassifier(loss='log', n_iter=n_iter, penalty='l2')

    def fit_and_predict(self):
        print "****** train start ******"
        print "sample:", list(self.y_train).count(0), list(self.y_train).count(1)
        print "train percent is %s" % (float(list(self.y_train).count(1)) / len(list(self.y_train)))
        start_at = time.time()

        x_train_features = self.scaler.fit_transform(self.x_train)
        # x_train_features = self.x_train
        self.classifier.fit(x_train_features, self.y_train)

        print "train done. score %s. spend %s s" \
              % (self.classifier.score(
                  x_train_features, self.y_train), time.time() - start_at)

        x_test_features = self.scaler.fit_transform(self.x_test)
        # x_test_features = self.x_test
        y_test_pred = self.classifier.predict(x_test_features)
        print "****** test result *******"
        print y_test_pred, len(y_test_pred)
        print list(y_test_pred).count(0), list(y_test_pred).count(1),\
            set(y_test_pred), len(set(y_test_pred))
        print "test percent is %s" % (float(list(y_test_pred).count(1)) / len(list(y_test_pred)))
        print accuracy_score(self.y_test, y_test_pred)

        x_pred_features = self.scaler.fit_transform(self.p_x)
        # x_pred_features = self.p_x
        y_pred = self.classifier.predict(x_pred_features)
        print "****** predict result *******"
        print y_pred, len(y_pred)
        print list(y_pred).count(0), list(y_pred).count(1), set(y_pred), len(set(y_pred))
        print "predict percent is %s" % (float(list(y_pred).count(1)) / len(list(y_pred)))

        return y_pred
