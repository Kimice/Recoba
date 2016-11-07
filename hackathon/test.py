# -*- coding: utf-8 -*-

import sys
import time
import math
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    Imputer,
    OneHotEncoder,
    LabelEncoder,
)

from predict import PredictWorker


print "***** train build start *****"
start_at = time.time()

FILE_PATH = str(sys.argv[1]).strip().rstrip('/')

his_eco_info = pd.DataFrame(pd.read_table('%s/his_eco_info.txt' % FILE_PATH))
his_eco_env = pd.DataFrame(pd.read_table('%s/his_eco_env.txt' % FILE_PATH))
# his_order_info = pd.DataFrame(pd.read_table('%s/his_order_info.txt' % FILE_PATH))
rst_info = pd.DataFrame(pd.read_table('%s/rst_info.txt' % FILE_PATH))

data = pd.merge(his_eco_env, his_eco_info, on='list_id')

cols = data.columns.tolist()
cols = cols[-3:-5:-1] + cols[:-4] + cols[-2:]
data = data[cols]
data = pd.merge(data, rst_info, on='restaurant_id')

data.drop(['list_id', 'is_raw_buy', 'order_id', 'eleme_device_id', 'x_x', 'x_y',
           'log_id', 'food_name_list', 'category_list', 'y_x', 'y_y'],
          axis=1, inplace=True)

le = LabelEncoder()
for param in ['brand', 'resolution', 'primary_category', 'bu_flag', 'user_id',
              'restaurant_id', 'brand_name', 'address_type', 'channel',
              'network_type', 'platform', 'model', 'network_operator']:
    data[param] = le.fit_transform(data[param])

ohe = OneHotEncoder()
for param in ['brand', 'resolution', 'primary_category', 'bu_flag', 'user_id',
              'restaurant_id', 'brand_name', 'address_type', 'channel',
              'network_type', 'platform', 'model', 'network_operator']:
    data[param] = le.fit_transform(data[param])

# data['good_rating_rate'] = data.apply(
#     lambda row: abs(row['good_rating_rate']),
#     axis=1
# )

print data.info()

x_click = np.array(data.iloc[:, 2:])
print x_click.shape
y_click = np.array(data.iloc[:, 1])

sel = VarianceThreshold(threshold=0.2)
x_click = sel.fit_transform(x_click)
print x_click.shape

# x_click = x_click.clip(min=0)
# skb = SelectKBest(chi2, k=10)
# x_click = skb.fit_transform(x_click, y_click)
# print x_click.shape
clf = ExtraTreesClassifier()
clf = clf.fit(x_click, y_click)
print clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
x_click = model.transform(x_click)
print x_click.shape

# data['distance'] = data.apply(
#     lambda row: math.sqrt(
#         (row['x_x'] - row['x_y'])**2 + (row['y_x'] - row['y_y'])**2
#     ),
#     axis=1
# )

# for column in data.columns.tolist():
#     print column


print "***** predict start *****"
next_eco_info = pd.DataFrame(pd.read_table('%s/next_eco_info.txt' % FILE_PATH))
next_eco_env = pd.DataFrame(pd.read_table('%s/next_eco_env.txt' % FILE_PATH))

p_data = pd.merge(next_eco_env, next_eco_info, on='list_id')
p_data = pd.merge(p_data, rst_info, on='restaurant_id')
#
# log_ids = list(p_data['log_id'])
#
# p_data['distance'] = p_data.apply(
#     lambda row: math.sqrt(
#         (row['x_x'] - row['x_y'])**2 + (row['y_x'] - row['y_y'])**2
#     ),
#     axis=1
# )
p_data.drop(['list_id', 'eleme_device_id', 'x_x', 'x_y',
            'log_id', 'food_name_list', 'category_list', 'y_x', 'y_y'],
            axis=1, inplace=True)

for param in ['brand', 'resolution', 'primary_category', 'bu_flag', 'user_id',
              'restaurant_id', 'brand_name', 'address_type', 'channel',
              'network_type', 'platform', 'model', 'network_operator']:
    p_data[param] = le.fit_transform(p_data[param])

print p_data.info()

x_pred = np.array(p_data.iloc[:, :])
print x_pred.shape

x_pred = sel.transform(x_pred)
print x_pred.shape

# x_pred = x_click.clip(min=0)
# x_pred = skb.transform(x_pred)
x_pred = model.transform(x_pred)
print x_pred.shape


#
# data.to_csv('files/data.csv')
# p_data.to_csv('files/p_data.csv')

# data.drop(data.columns[0], axis=1, inplace=True)
# p_data.drop(p_data.columns[0], axis=1, inplace=True)

# le = LabelEncoder()
# data['bu_flag'] = le.fit_transform(data['bu_flag'].values)
# data['user_id'] = le.fit_transform(data['user_id'].values)
# data['restaurant_id'] = le.fit_transform(data['restaurant_id'].values)
# p_data['bu_flag'] = le.fit_transform(p_data['bu_flag'].values)
# p_data['user_id'] = le.fit_transform(p_data['user_id'].values)
# p_data['restaurant_id'] = le.fit_transform(p_data['restaurant_id'].values)

# x_click = np.array(data.iloc[:, 2:])
# y_click = np.array(data.iloc[:, 1])
# x_buy = np.array(data.iloc[:, 2:])  # not include is_click
# y_buy = np.array(data.iloc[:, 0])  # is_buy

# data.drop(['is_click', 'is_buy'], axis=1, inplace=True)

# print data.info(), p_data.info()

# pca = PCA(n_components=1)
# pca.fit(data)
# print pca.components_
# print pca.explained_variance_ratio_
# data = pd.DataFrame(pca.fit_transform(data))
# p_data = pd.DataFrame(pca.fit_transform(p_data))


# data = pd.merge(data, his_order_info, on='order_id')
# data = pd.DataFrame(Imputer().fit_transform(data.values))
# p_data = pd.DataFrame(Imputer().fit_transform(p_data.values))
# print "build spend %s s." % (time.time() - start_at)
# print "***** analysis start *****"


print "sample is_click:", list(y_click).count(0), list(y_click).count(1)
# x_click = np.array(data)
y_pred = PredictWorker(x_click, y_click, x_pred).fit_and_predict()


# print "sample is_buy:", list(y_buy).count(0), list(y_buy).count(1)
# y_buy_pred = fit_and_predict(x_buy, y_buy, x_pred)



