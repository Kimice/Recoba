# -*- coding: utf-8 -*-

import sys
import time
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer

from predict import PredictWorker

print "***** train build start *****"
start_at = time.time()

FILE_PATH = str(sys.argv[1]).strip().rstrip('/')

his_eco_info = pd.DataFrame(pd.read_table('%s/his_eco_info.txt' % FILE_PATH))
his_eco_env = pd.DataFrame(pd.read_table('%s/his_eco_env.txt' % FILE_PATH))
his_order_info = pd.DataFrame(pd.read_table('%s/his_order_info.txt' % FILE_PATH))
rst_info = pd.DataFrame(pd.read_table('%s/rst_info.txt' % FILE_PATH))

data = pd.merge(his_eco_env, his_eco_info, on='list_id')

cols = data.columns.tolist()
cols = cols[-3:-5:-1] + cols[:-4] + cols[-2:]
data = data[cols]
data = pd.merge(data, rst_info, on='restaurant_id')

# ------ calculate distance
data['distance'] = data.apply(
    lambda row: math.sqrt(
        (row['x_x'] - row['x_y'])**2 + (row['y_x'] - row['y_y'])**2
    ),
    axis=1
)

data.drop(['list_id', 'is_raw_buy', 'order_id', 'eleme_device_id', 'x_x', 'x_y', 'user_id',
           'network_type', 'platform', 'brand', 'model', 'network_operator', 'resolution',
           'channel', 'log_id', 'restaurant_id', 'primary_category', 'food_name_list',
           'category_list', 'y_x', 'y_y', 'address_type', 'bu_flag', 'brand_name'],
          axis=1, inplace=True)

# data = pd.merge(data, his_order_info, on='order_id')
data = pd.DataFrame(Imputer().fit_transform(data.values))
print "build spend %s s." % (time.time() - start_at)

print "***** analysis start *****"
start_at = time.time()

x_click = np.array(data.iloc[:, 2:])
y_click = np.array(data[1])  # is_click
x_buy = np.array(data.iloc[:, 2:])  # not include is_click
y_buy = np.array(data[0])  # is_buy

print "sample is_click:", list(y_click).count(0), list(y_click).count(1)
print "sample is_buy:", list(y_buy).count(0), list(y_buy).count(1)

print "***** predict start *****"
next_eco_info = pd.DataFrame(pd.read_table('%s/next_eco_info.txt' % FILE_PATH))
next_eco_env = pd.DataFrame(pd.read_table('%s/next_eco_env.txt' % FILE_PATH))

p_data = pd.merge(next_eco_env, next_eco_info, on='list_id')
p_data = pd.merge(p_data, rst_info, on='restaurant_id')

log_ids = list(p_data['log_id'])

# ------ calculate distance
p_data['distance'] = p_data.apply(
    lambda row: math.sqrt(
        (row['x_x'] - row['x_y'])**2 + (row['y_x'] - row['y_y'])**2
    ),
    axis=1
)
p_data.drop(['list_id', 'eleme_device_id', 'x_x', 'x_y', 'user_id', 'network_type',
             'platform', 'brand', 'model', 'network_operator', 'resolution',
             'channel', 'log_id', 'restaurant_id', 'primary_category', 'food_name_list',
             'category_list', 'y_x', 'y_y', 'address_type', 'bu_flag', 'brand_name'],
            axis=1, inplace=True
            )
p_data = pd.DataFrame(Imputer().fit_transform(p_data.values))

x_pred = np.array(p_data)
y_click_pred = PredictWorker(x_click, y_click, x_pred).fit_and_predict()
y_buy_pred = PredictWorker(x_buy, y_buy, x_pred).fit_and_predict()

with open('/output/result.txt', 'w+') as f:
    for i in xrange(len(log_ids)):
        f.write('   '.join(
            (
                str(log_ids[i]),
                str(int(y_click_pred[i])),
                str(int(y_buy_pred[i]))
            )
        ) + '\n')




