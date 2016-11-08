# -*- coding: utf-8 -*-

import sys
import math
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    Imputer
)

from predict import PredictWorker

file_path = str(sys.argv[1]).strip().rstrip('/')
his_eco_info = pd.DataFrame(pd.read_table('%s/his_eco_info.txt' % file_path))
his_eco_env = pd.DataFrame(pd.read_table('%s/his_eco_env.txt' % file_path))
his_order_info = pd.DataFrame(pd.read_table('%s/his_order_info.txt' % file_path))
rst_info = pd.DataFrame(pd.read_table('%s/rst_info.txt' % file_path))
next_eco_info = pd.DataFrame(pd.read_table('%s/next_eco_info.txt' % file_path))
next_eco_env = pd.DataFrame(pd.read_table('%s/next_eco_env.txt' % file_path))


def random_sample(data, n):
    return data.ix[random.sample(data.index, n)]


def encode(data):
    le = LabelEncoder()
    for param in ['brand', 'resolution', 'primary_category', 'address_type',
                  'network_type', 'platform', 'model', 'network_operator',
                  'channel', 'brand_name', 'bu_flag']:
        data[param] = le.fit_transform(data[param])


def calc_rst_recent_status(restaurant_id, day_no,
                           is_time_ensure, time_ensure_spent):
    """

    :param restaurant_id:
    :param day_no:
    :return: valid_order_count, invalid_order_count, invoice_order_count, coupon_order_count,
             pay_amount, dish_amount
    """
    valid_order_count, invalid_order_count,\
        invoice_order_count, coupon_order_count,\
        pay_amount, dish_amount, time_ensure,\
        restaurant_order_count \
        = 0, 0, 0, 0, 0, 0, 0, 0
    df = his_order_info.loc[(his_order_info['restaurant_id'] == restaurant_id)]
    if not df.empty:
        restaurant_order_count = df.is_valid.sum()
        valid_order_count = df.is_valid.sum()
        invalid_order_count = df.loc[df['is_valid'] == 0].is_valid.count()
        invoice_order_count = df.loc[df['is_valid'] == 1].is_invoice.sum()
        coupon_order_count = df.loc[df['is_valid'] == 1].is_coupon.sum()
        pay_amount = df.loc[df['is_valid'] == 1].eleme_order_total.sum()
        dish_amount = df.loc[df['is_valid'] == 1].total.sum()
        time_ensure = time_ensure_spent if is_time_ensure else 120
    return valid_order_count, invalid_order_count, invoice_order_count,\
        coupon_order_count, pay_amount, dish_amount, time_ensure,\
        restaurant_order_count


def del_data(data):
    print data.info()
    # data['distance'] = data.apply(
    #     lambda row: math.sqrt(
    #         (row['x_x'] - row['x_y'])**2 + (row['y_x'] - row['y_y'])**2
    #     ),
    #     axis=1
    # )

    data.drop(['x_x', 'x_y', 'y_x', 'y_y'],
              axis=1, inplace=True)

    # data = data.loc[(data['distance'] <= data['radius'])]
    # data.drop(['distance', 'radius'],
    #           axis=1, inplace=True)

    # data[['valid_order_count', 'invalid_order_count', 'invoice_order_count',
    #       'coupon_order_count', 'pay_amount', 'dish_amount', 'time_ensure',
    #       'restaurant_order_count']] = data.apply(
    #     lambda row: calc_rst_recent_status(
    #         row['restaurant_id'], row['day_no'],
    #         row['is_time_ensure'], row['time_ensure_spent']
    #     ),
    #     axis=1
    # ).apply(pd.Series)
    data.drop(['day_no', 'log_id', 'restaurant_id', 'invoice',
               'is_time_ensure', 'time_ensure_spent', 'is_time_ensure_discount'],
              axis=1, inplace=True)

    encode(data)

    # data = data.loc[(data[''] == 1)]
    # print data.info()
    data = pd.DataFrame(Imputer().fit_transform(data))
    return data


def run():
    data = pd.merge(his_eco_env, his_eco_info, on='list_id')

    cols = data.columns.tolist()
    cols = cols[-3:-5:-1] + cols[:-4] + cols[-2:]
    data = data[cols]
    data = pd.merge(data, rst_info, on='restaurant_id')

    data = random_sample(data, 20000)

    data.drop(['list_id', 'is_raw_buy', 'order_id', 'eleme_device_id',
               'food_name_list', 'category_list'],
              axis=1, inplace=True)

    # anonymous_data = data.loc[pd.isnull(data['user_id'])]
    # anonymous_data.drop(['user_id'], axis=1, inplace=True)
    # anonymous_data = del_data(anonymous_data)
    # x_click = np.array(anonymous_data.iloc[:, 2:])
    # y_click = np.array(anonymous_data.iloc[:, 1])
    # p_click = PredictWorker(x_click, y_click).fit_and_predict()
    # print p_click

    known_data = data.loc[pd.notnull(data['user_id'])]
    known_data.drop(['user_id'], axis=1, inplace=True)
    known_data = del_data(known_data)
    x_click = np.array(known_data.iloc[:, 2:])
    y_click = np.array(known_data.iloc[:, 1])
    p_click = PredictWorker(x_click, y_click).fit_and_predict()

    # p_data = pd.merge(next_eco_env, next_eco_info, on='list_id')
    # p_data = pd.merge(p_data, rst_info, on='restaurant_id')
    # p_data.drop(['list_id', 'eleme_device_id',
    #             'food_name_list', 'category_list'],
    #             axis=1, inplace=True)

if __name__ == '__main__':
    run()
