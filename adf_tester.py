import dataloader_KAMIS
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def preprocess_prices(prices):
    return list(map(lambda x: float(x.replace(',', '')), prices))

# Preprocess Seoul data
seoul_apple = dataloader_KAMIS.seoul_apple
seoul_apple_price = preprocess_prices(seoul_apple['price'])
seoul_baechu = dataloader_KAMIS.seoul_baechu
seoul_baechu_price = preprocess_prices(seoul_baechu['price'])
seoul_fish = dataloader_KAMIS.seoul_fish
seoul_fish_price = preprocess_prices(seoul_fish['price'])
seoul_pig = dataloader_KAMIS.seoul_pig
seoul_pig_price = preprocess_prices(seoul_pig['price'])
seoul_rice = dataloader_KAMIS.seoul_rice
seoul_rice_price = preprocess_prices(seoul_rice['price'])

# Preprocess Busan data
busan_apple = dataloader_KAMIS.busan_apple
busan_apple_price = preprocess_prices(busan_apple['price'])
busan_baechu = dataloader_KAMIS.busan_baechu
busan_baechu_price = preprocess_prices(busan_baechu['price'])
busan_fish = dataloader_KAMIS.busan_fish
busan_fish_price = preprocess_prices(busan_fish['price'])
busan_pig = dataloader_KAMIS.busan_pig
busan_pig_price = preprocess_prices(busan_pig['price'])
busan_rice = dataloader_KAMIS.busan_rice
busan_rice_price = preprocess_prices(busan_rice['price'])

# Preprocess Daegu data
daegu_apple = dataloader_KAMIS.daegu_apple
daegu_apple_price = preprocess_prices(daegu_apple['price'])
daegu_baechu = dataloader_KAMIS.daegu_baechu
daegu_baechu_price = preprocess_prices(daegu_baechu['price'])
daegu_fish = dataloader_KAMIS.daegu_fish
daegu_fish_price = preprocess_prices(daegu_fish['price'])
daegu_pig = dataloader_KAMIS.daegu_pig
daegu_pig_price = preprocess_prices(daegu_pig['price'])
daegu_rice = dataloader_KAMIS.daegu_rice
daegu_rice_price = preprocess_prices(daegu_rice['price'])

def adf_test(data, title):
    result = adfuller(data)
    print(f'ADF Statistics for {title}: %f' % result[0])
    print('p-value: %f' % result[1])
    print('num of lags: %f' % result[2])
    print('num of observations: %f' % result[3])
    print('Critical values:')
    for k, v in result[4].items():
        print('\t%s: %.3f' % (k,v))
    print('\n')

def adf_test_diff1(data, title):
    data = pd.Series(data).diff().dropna()
    result = adfuller(data)
    print(f'ADF Diff 1 Statistics for {title}: %f' % result[0])
    print('p-value: %f' % result[1])
    print('num of lags: %f' % result[2])
    print('num of observations: %f' % result[3])
    print('Critical values:')
    for k, v in result[4].items():
        print('\t%s: %.3f' % (k,v))
    print('\n')

def adf_test_diff2(data, title):
    data = pd.Series(data).diff().diff().dropna()
    result = adfuller(data)
    print(f'ADF Diff 2 Statistics for {title}: %f' % result[0])
    print('p-value: %f' % result[1])
    print('num of lags: %f' % result[2])
    print('num of observations: %f' % result[3])
    print('Critical values:')
    for k, v in result[4].items():
        print('\t%s: %.3f' % (k,v))
    print('\n')

adf_test(seoul_apple_price, 'Seoul Apple Price')
adf_test(seoul_baechu_price, 'Seoul Baechu Price')
adf_test(seoul_fish_price, 'Seoul Fish Price')
adf_test(seoul_pig_price, 'Seoul Pig Price')
adf_test(seoul_rice_price, 'Seoul Rice Price')

adf_test(busan_apple_price, 'Busan Apple Price')
adf_test(busan_baechu_price, 'Busan Baechu Price')
adf_test(busan_fish_price, 'Busan Fish Price')
adf_test(busan_pig_price, 'Busan Pig Price')
adf_test(busan_rice_price, 'Busan Rice Price')

adf_test(daegu_apple_price, 'Daegu Apple Price')
adf_test(daegu_baechu_price, 'Daegu Baechu Price')
adf_test(daegu_fish_price, 'Daegu Fish Price')
adf_test(daegu_pig_price, 'Daegu Pig Price')
adf_test(daegu_rice_price, 'Daegu Rice Price')

adf_test_diff1(seoul_apple_price, 'Seoul Apple Price')
adf_test_diff1(seoul_baechu_price, 'Seoul Baechu Price')
adf_test_diff1(seoul_fish_price, 'Seoul Fish Price')
adf_test_diff1(seoul_pig_price, 'Seoul Pig Price')
adf_test_diff1(seoul_rice_price, 'Seoul Rice Price')

adf_test_diff1(busan_apple_price, 'Busan Apple Price')
adf_test_diff1(busan_baechu_price, 'Busan Baechu Price')
adf_test_diff1(busan_fish_price, 'Busan Fish Price')
adf_test_diff1(busan_pig_price, 'Busan Pig Price')
adf_test_diff1(busan_rice_price, 'Busan Rice Price')

adf_test_diff1(daegu_apple_price, 'Daegu Apple Price')
adf_test_diff1(daegu_baechu_price, 'Daegu Baechu Price')
adf_test_diff1(daegu_fish_price, 'Daegu Fish Price')
adf_test_diff1(daegu_pig_price, 'Daegu Pig Price')
adf_test_diff1(daegu_rice_price, 'Daegu Rice Price')

adf_test_diff2(seoul_apple_price, 'Seoul Apple Price')
adf_test_diff2(seoul_baechu_price, 'Seoul Baechu Price')
adf_test_diff2(seoul_fish_price, 'Seoul Fish Price')
adf_test_diff2(seoul_pig_price, 'Seoul Pig Price')
adf_test_diff2(seoul_rice_price, 'Seoul Rice Price')

adf_test_diff2(busan_apple_price, 'Busan Apple Price')
adf_test_diff2(busan_baechu_price, 'Busan Baechu Price')
adf_test_diff2(busan_fish_price, 'Busan Fish Price')
adf_test_diff2(busan_pig_price, 'Busan Pig Price')
adf_test_diff2(busan_rice_price, 'Busan Rice Price')

adf_test_diff2(daegu_apple_price, 'Daegu Apple Price')
adf_test_diff2(daegu_baechu_price, 'Daegu Baechu Price')
adf_test_diff2(daegu_fish_price, 'Daegu Fish Price')
adf_test_diff2(daegu_pig_price, 'Daegu Pig Price')
adf_test_diff2(daegu_rice_price, 'Daegu Rice Price')

# adf_test(data)
# # Original p-value = 0.07 > 0.05 -> Fail to reject null hypothesis

# dff1 = data.diff().dropna()
# adf_test(dff1)
# First derivative p-value = 0.00 < 0.05 -> First order stationary

# adf_test(data2)
# # Original p-value = 0.26 > 0.05 -> Fail to reject null hypothesis

# dff2 = data2.diff().dropna()
# adf_test(dff2)
# # First derivative p-value = 0.00 < 0.05 -> First order stationary

# plt.figure(figsize=(15, 5))
# plt.plot(dff1)
# plt.show()