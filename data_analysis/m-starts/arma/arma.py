import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

feature_names = ['year', 'count']
data_set = pd.read_csv('../m-Starts.txt', sep=' ', names=feature_names)
data_set.index = pd.Index(sm.tsa.datetools.dates_from_range('1959m1', '2010m2'))
# print data_set.info()
# print data_set.describe()

del data_set['year']
data_set.plot()
plt.savefig('origin_data.png', dip=100)
plt.show()

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
diff1 = data_set.diff(2)
diff1.plot(ax=ax1)
plt.savefig('diff1.png', dip=100)
plt.show()

temp = np.array(data_set)[1:]
# fig = plt.figure(figsize=(20, 10))
# ax1=fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(temp,lags=30,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(temp,lags=30,ax=ax2)

# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(data_set.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(data_set, lags=40, ax=ax2)
# plt.savefig('arma.png', dpi=100)
# plt.show()

print sm.tsa.arma_order_select_ic(temp,max_ar=6,max_ma=4,ic='aic')['aic_min_order']  # AIC

arma_mod10 = sm.tsa.ARMA(data_set[1:], (5, 4)).fit()
plt.figure(figsize=(15,5))
plt.plot(arma_mod10.fittedvalues,label='fitted value')
plt.plot(temp,label='real value')
plt.legend(loc=0)
plt.savefig('compare.png', dpi=100)
plt.show()

resid = arma_mod10.resid
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.savefig('qq.png', dpi=100)
plt.show()

# predict_sunspots = arma_mod10.predict('2010m1', '2100m12', dynamic=True)
# # predict_sunspots = arma_mod10.forecast(steps=1)
# # print(predict_sunspots)
# fig, ax = plt.subplots(figsize=(12, 8))
# # ax = data_set['2001m1':].plot(ax=ax)
# predict_sunspots.plot(ax=ax)
# plt.show()
