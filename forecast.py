import dataloader
data = dataloader.gwangjang_apple_df
data = data['가격(원)']

data2 = dataloader.DACON_GARLIC

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pmdarima.arima import ndiffs

def forecast_n_step(model, n=1):
    fc, conf_int = model.predict(n_periods=n, return_conf_int=True)
    return (
        fc.tolist()[0:n], np.asarray(conf_int).tolist()[0:n]
    )

def forecast(len, model, index, data=None):
    y_pred = []
    pred_upper = []
    pred_lower = []

    if data is not None:
        for new_ob in data:
            fc, conf = forecast_n_step(model)
            y_pred.append(fc[0])
            pred_upper.append(conf[0][1])
            pred_lower.append(conf[0][0])
            model.update(new_ob)
    else:
        for i in range(len):
            fc, conf = forecast_n_step(model)
            y_pred.append(fc[0])
            pred_upper.append(conf[0][1])
            pred_lower.append(conf[0][0])
            model.update(fc[0])
    return pd.Series(y_pred, index=index), pred_upper, pred_lower

# Fit the model for data2
model_fit_garlic = pm.auto_arima(
    y=data2, 
    d=1, 
    start_p=0, max_p=2, 
    start_q=0, max_q=2, 
    m=1, seasonal=False, 
    stepwise=True,
    trace=True
)

# Forecast
n_future = 10  # 예측할 미래 데이터의 개수
fc2, upper, lower = forecast(len(data2), model_fit_garlic, data2.index, data=data2)

# 미래 예측
future_index = pd.date_range(start=data2.index[-1], periods=n_future + 1, freq='M')[1:]
future_forecast, future_upper, future_lower = forecast(n_future, model_fit_garlic, future_index)

# pandas series 생성
fc_series = pd.Series(fc2, index=data2.index)  # 예측결과
lower_series = pd.Series(lower, index=data2.index)  # 예측결과의 하한 바운드
upper_series = pd.Series(upper, index=data2.index)  # 예측결과의 상한 바운드

future_fc_series = pd.Series(future_forecast, index=future_index)
future_lower_series = pd.Series(future_lower, index=future_index)
future_upper_series = pd.Series(future_upper, index=future_index)

# Plot
plt.figure(figsize=(20,6))
plt.plot(data2, label='Actual Data')  # 실제 데이터
plt.plot(fc_series, c='r', label='Predicted Price')  # 예측 데이터
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)  # 예측 범위

# 미래 예측 데이터와 범위를 기존 데이터 오른쪽에 이어서 표시
plt.plot(future_fc_series, c='b', label='Future Predicted Price')  # 미래 예측 데이터
plt.fill_between(future_lower_series.index, future_lower_series, future_upper_series, color='c', alpha=.10)  # 미래 예측 범위

plt.axvline(x=data2.index[-1], color='k', linestyle='--')  # 현재 데이터와 미래 예측 데이터 경계선 표시

plt.legend(loc='upper left')
plt.show()
