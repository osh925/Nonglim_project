import dataloader
data = dataloader.gwangjang_apple_df
data = data['가격(원)']

data2 = dataloader.DACON_GARLIC

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pmdarima.arima import ndiffs

n_diffs = ndiffs(data, alpha=0.05, test='adf', max_d=6)
print(f"추정된 차수 d = {n_diffs}") # 결과

n_diffs2 = ndiffs(data2, alpha=0.05, test='adf', max_d=6)
print(f"추정된 차수 d = {n_diffs2}") # 결과

model = pm.auto_arima(
            y=data, 
            d=1, 
            start_p=0, max_p=3, 
            start_q=0, max_q=3, 
            m=1, seasonal=False,
            stepwise=True,
            trace=True
)

# 광장 애플은 아리마 112가 적절

model = pm.auto_arima(
            y=data2, 
            d=1, 
            start_p=0, max_p=3, 
            start_q=0, max_q=3, 
            m=1, seasonal=False,
            stepwise=True,
            trace=True
)

# DACON 데이터는 아리마 211이 적절
## 왜 그런지는 ㅅㅂ 모르겠음