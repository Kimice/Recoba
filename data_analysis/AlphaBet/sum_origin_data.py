# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd

INTER_COLS = ['AwayTeam', 'B365A', 'B365D', 'B365H', 'BWA', 'BWD', 'BWH',
              'Bb1X2', 'BbAH', 'BbAHh', 'BbAv<2.5', 'BbAv>2.5', 'BbAvA', 'BbAvAHA', 'BbAvAHH', 'BbAvD',
              'BbAvH', 'BbMx<2.5', 'BbMx>2.5', 'BbMxA', 'BbMxAHA', 'BbMxAHH', 'BbMxD', 'BbMxH', 'BbOU',
              'Date', 'Div', 'FTAG', 'FTHG', 'FTR', 'HTAG', 'HTHG', 'HTR', 'HomeTeam',
              'IWA', 'IWD', 'IWH', 'LBA', 'LBD', 'LBH', 'VCA', 'VCD', 'VCH', 'WHA', 'WHD', 'WHH']

UNION_COLS = ['AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'AwayTeam',
              'B365<2.5', 'B365>2.5', 'B365A', 'B365AH', 'B365AHA', 'B365AHH', 'B365D', 'B365H',
              'BSA', 'BSD', 'BSH', 'BWA', 'BWD', 'BWH', 'Bb1X2', 'BbAH', 'BbAHh',
              'BbAv<2.5', 'BbAv>2.5', 'BbAvA', 'BbAvAHA', 'BbAvAHH', 'BbAvD', 'BbAvH',
              'BbMx<2.5', 'BbMx>2.5', 'BbMxA', 'BbMxAHA', 'BbMxAHH', 'BbMxD', 'BbMxH', 'BbOU',
              'Date', 'Div', 'FTAG', 'FTHG', 'FTR', 'GB<2.5', 'GB>2.5', 'GBA', 'GBAH', 'GBAHA', 'GBAHH',
              'GBD', 'GBH', 'HC', 'HF', 'HR', 'HS', 'HST', 'HTAG', 'HTHG', 'HTR', 'HY',
              'HomeTeam', 'IWA', 'IWD', 'IWH', 'LBA', 'LBAH', 'LBAHA', 'LBAHH', 'LBD', 'LBH',
              'PSA', 'PSCA', 'PSCD', 'PSCH', 'PSD', 'PSH', 'Referee', 'SBA', 'SBD', 'SBH',
              'SJA', 'SJD', 'SJH', 'SOA', 'SOD', 'SOH', 'VCA', 'VCD', 'VCH', 'WHA', 'WHD', 'WHH']

USEFUL_COLS = [
    'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',
    'LBH', 'LBD', 'LBA', 'VCH', 'VCD', 'VCA', 'WHH', 'WHD', 'WHA',
    'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA',
    'BbOU', 'BbMx>2.5', 'BbMx<2.5', 'BbAv>2.5', 'BbAv<2.5',
    'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA'
]


def union_or_inter(path):
    cols = set(UNION_COLS)
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                data = pd.DataFrame(pd.read_csv(path + '/' + f))
                d_cols = data.columns
                if 'Bb1X2' in d_cols:
                    is_add = True
                    for i in d_cols:
                        if i.startswith('Unnamed'):
                            is_add = False
                            continue
                    if is_add:
                        cols = cols.intersection(data.columns)
            except Exception as e:
                print f, e.message
    print sorted(cols), len(cols)


def work(path):
    odds_data = pd.DataFrame(columns=USEFUL_COLS)
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                data = pd.DataFrame(pd.read_csv(path + '/' + f))
                d_cols = data.columns
                if 'BbMx>2.5' in d_cols:
                    data = pd.DataFrame(pd.read_csv(path + '/' + f), columns=USEFUL_COLS)
                    odds_data = odds_data.append(data, ignore_index=True)
            except Exception as e:
                print f, e.message
    print odds_data.info(), odds_data.shape
    odds_data.to_csv('datas/odds.csv')

if __name__ == '__main__':
    # work('/Users/kimice/Downloads/odds')
    union_or_inter('/Users/kimice/Downloads/odds')
