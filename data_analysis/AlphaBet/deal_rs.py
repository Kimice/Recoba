# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import seaborn as sns


def deal_rs():
    data_set = pd.read_csv('datas/result.csv')
    data_set.columns = ['AvH', 'AvD', 'AvA', 'Hc', 'Dc', 'Ac', 'R', 'P']

    sns.set(style='ticks')
    sns.lmplot(x='R', y='P', data=data_set)
    sns.plt.show()


if __name__ == '__main__':
    deal_rs()



