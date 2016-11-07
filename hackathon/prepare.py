# -*- coding: utf-8 -*-

import sys
import time
import math
import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


class PrepareWorker(object):
    def __init__(self, file_path, rand_num=200000):
        self.file_path = str(sys.argv[1]).strip().rstrip('/')
        self.rand_num = rand_num
        self.his_eco_info = pd.DataFrame(pd.read_table('%s/his_eco_info.txt' % file_path))
        self.his_eco_env = pd.DataFrame(pd.read_table('%s/his_eco_env.txt' % file_path))
        self.his_order_info = pd.DataFrame(pd.read_table('%s/his_order_info.txt' % file_path))
        self.rst_info = pd.DataFrame(pd.read_table('%s/rst_info.txt' % file_path))
        self.next_eco_info = pd.DataFrame(pd.read_table('%s/next_eco_info.txt' % file_path))
        self.next_eco_env = pd.DataFrame(pd.read_table('%s/next_eco_env.txt' % file_path))

    def random_sample(self, data, n):
        return data.ix[random.sample(data.index, n)]

    def encode(self, data):
        le = LabelEncoder()
        for param in ['brand', 'resolution', 'primary_category',
                      'network_type', 'platform', 'model', 'network_operator',
                      'channel', 'brand_name', 'address_type', 'bu_flag']:
            data[param] = le.fit_transform(data[param])

    def calc_order_recent_status(self, user_id, restaurant_id, day_no,
                                 is_time_ensure, time_ensure_spent):
        """

        :param user_id:
        :param restaurant_id:
        :param day_no:
        :return: valid_order_count, invalid_order_count, invoice_order_count, coupon_order_count,
                 pay_amount, dish_amount
        """
        valid_order_count, invalid_order_count,\
            invoice_order_count, coupon_order_count,\
            pay_amount, dish_amount, time_ensure = 0, 0, 0, 0, 0, 0, 0
        df = self.his_order_info.loc[(self.his_order_info['user_id'] == user_id) &
                                     (self.his_order_info['restaurant_id'] == restaurant_id) &
                                     (self.his_order_info['day_no'] < day_no) &
                                     (self.his_order_info['day_no'] >= day_no - 14) &
                                     (self.his_order_info['has_new_user_subsidy'] == 0)]
        if not df.empty:
            valid_order_count = df.is_valid.sum()
            invalid_order_count = df.loc[df['is_valid'] == 0].is_valid.count()
            invoice_order_count = df.loc[df['is_valid'] == 1].is_invoice.sum()
            coupon_order_count = df.loc[df['is_valid'] == 1].is_coupon.sum()
            pay_amount = df.loc[df['is_valid'] == 1].eleme_order_total.sum()
            dish_amount = df.loc[df['is_valid'] == 1].total.sum()
            time_ensure = time_ensure_spent if is_time_ensure else 120
        return valid_order_count, invalid_order_count, invoice_order_count,\
            coupon_order_count, pay_amount, dish_amount, time_ensure

    def prepare_click_train_data(self):
        print "***** data build start *****"
        start_at = time.time()

        data = pd.merge(self.his_eco_env, self.his_eco_info, on='list_id')

        cols = data.columns.tolist()
        cols = cols[-3:-5:-1] + cols[:-4] + cols[-2:]
        data = data[cols]
        data = pd.merge(data, self.rst_info, on='restaurant_id')
        has_click = data.loc[(data['is_click'] == 1)]
        no_click = data.loc[(data['is_click'] == 0)]
        no_click = self.random_sample(no_click, int(len(has_click) * 2.42))
        data = has_click.append(no_click)
        data.drop(['list_id', 'is_raw_buy', 'order_id', 'eleme_device_id',
                   'food_name_list', 'category_list'],
                  axis=1, inplace=True)

        data = data.loc[(data['day_no'] <= 140)]
        data['distance'] = data.apply(
            lambda row: math.sqrt(
                (row['x_x'] - row['x_y'])**2 + (row['y_x'] - row['y_y'])**2
            ),
            axis=1
        )

        data.drop(['x_x', 'x_y', 'y_x', 'y_y'],
                  axis=1, inplace=True)

        self.encode(data)

        # data[['valid_order_count', 'invalid_order_count', 'invoice_order_count',
        #       'coupon_order_count', 'pay_amount', 'dish_amount', 'time_ensure']] = data.apply(
        #     lambda row: self.calc_order_recent_status(
        #         row['user_id'], row['restaurant_id'], row['day_no'],
        #         row['is_time_ensure'], row['time_ensure_spent']
        #     ),
        #     axis=1
        # ).apply(pd.Series)
        data.drop(['day_no', 'user_id', 'log_id', 'restaurant_id', 'invoice',
                   'is_time_ensure', 'time_ensure_spent', 'is_time_ensure_discount'],
                  axis=1, inplace=True)

        return data

    def prepare_buy_train_data(self):
        print "***** data build start *****"
        start_at = time.time()

        data = pd.merge(self.his_eco_env, self.his_eco_info, on='list_id')

        cols = data.columns.tolist()
        cols = cols[-3:-5:-1] + cols[:-4] + cols[-2:]
        data = data[cols]
        data = pd.merge(data, self.rst_info, on='restaurant_id')
        has_buy = data.loc[(data['is_buy'] == 1)]
        no_buy = data.loc[(data['is_buy'] == 0)]
        no_buy = self.random_sample(no_buy, int(len(has_buy) * 9.9))
        data = has_buy.append(no_buy)
        data.drop(['list_id', 'is_raw_buy', 'order_id', 'eleme_device_id',
                   'food_name_list', 'category_list'],
                  axis=1, inplace=True)

        data = data.loc[(data['day_no'] <= 140)]

        data['distance'] = data.apply(
            lambda row: math.sqrt(
                (row['x_x'] - row['x_y'])**2 + (row['y_x'] - row['y_y'])**2
            ),
            axis=1
        )

        data.drop(['x_x', 'x_y', 'y_x', 'y_y'],
                  axis=1, inplace=True)

        self.encode(data)

        # data[['valid_order_count', 'invalid_order_count', 'invoice_order_count',
        #       'coupon_order_count', 'pay_amount', 'dish_amount', 'time_ensure']] = data.apply(
        #     lambda row: self.calc_order_recent_status(
        #         row['user_id'], row['restaurant_id'], row['day_no'],
        #         row['is_time_ensure'], row['time_ensure_spent']
        #     ),
        #     axis=1
        # ).apply(pd.Series)
        data.drop(['day_no', 'user_id', 'log_id', 'restaurant_id', 'invoice',
                   'is_time_ensure', 'time_ensure_spent', 'is_time_ensure_discount'],
                  axis=1, inplace=True)

        data = data.iloc[np.random.permutation(len(data))]
        return data

    def prepare_predict_data(self):
        print "***** predict build start *****"
        start_at = time.time()

        data = pd.merge(self.next_eco_env, self.next_eco_info, on='list_id')
        data = pd.merge(data, self.rst_info, on='restaurant_id')

        log_ids = list(data['log_id'])

        data['distance'] = data.apply(
            lambda row: math.sqrt(
                (row['x_x'] - row['x_y'])**2 + (row['y_x'] - row['y_y'])**2
            ),
            axis=1
        )

        data.drop(['list_id', 'eleme_device_id', 'x_x', 'x_y',
                   'food_name_list', 'category_list', 'y_x', 'y_y'],
                  axis=1, inplace=True)

        self.encode(data)

        # data[['valid_order_count', 'invalid_order_count', 'invoice_order_count',
        #       'coupon_order_count', 'pay_amount', 'dish_amount', 'time_ensure']] = data.apply(
        #     lambda row: self.calc_order_recent_status(
        #         row['user_id'], row['restaurant_id'], row['day_no'],
        #         row['is_time_ensure'], row['time_ensure_spent']
        #     ),
        #     axis=1
        # ).apply(pd.Series)
        data.drop(['day_no', 'user_id', 'log_id', 'restaurant_id', 'invoice',
                   'is_time_ensure', 'time_ensure_spent', 'is_time_ensure_discount'],
                  axis=1, inplace=True)

        data = data.iloc[np.random.permutation(len(data))]
        return data, log_ids
