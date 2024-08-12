import dataloader_KAMIS
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima.arima import ndiffs

def preprocess_prices(prices):
    return list(map(lambda x: float(x.replace(',', '')), prices))

# Function to fit model and forecast
def fit_and_forecast(price_series, n_future=10):
    model_fit = pm.auto_arima(
        y=price_series, 
        d=1, 
        start_p=0, max_p=2, 
        start_q=0, max_q=2, 
        m=1, seasonal=False, 
        stepwise=True,
        trace=True
    )
    fc, upper, lower = forecast(len(price_series), model_fit, price_series.index, data=price_series)
    future_index = pd.date_range(start=price_series.index[-1], periods=n_future + 1, freq='M')[1:]
    future_forecast, future_upper, future_lower = forecast(n_future, model_fit, future_index)
    
    return fc, upper, lower, future_forecast, future_upper, future_lower, price_series.index, future_index

def plot_forecast(price_series, fc, upper, lower, future_forecast, future_upper, future_lower, future_index, title):
    fc_series = pd.Series(fc, index=price_series.index)
    lower_series = pd.Series(lower, index=price_series.index)
    upper_series = pd.Series(upper, index=price_series.index)
    future_fc_series = pd.Series(future_forecast, index=future_index)
    future_lower_series = pd.Series(future_upper, index=future_index)
    future_upper_series = pd.Series(future_lower, index=future_index)
    
    plt.figure(figsize=(20, 6))
    plt.plot(price_series, label='Actual Data')
    plt.plot(fc_series, c='r', label='Predicted Price')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
    plt.plot(future_fc_series, c='b', label='Future Predicted Price')
    plt.fill_between(future_lower_series.index, future_lower_series, future_upper_series, color='c', alpha=.10)
    plt.axvline(x=price_series.index[-1], color='k', linestyle='--')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

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
            model.update([new_ob])
    else:
        for i in range(len):
            fc, conf = forecast_n_step(model)
            y_pred.append(fc[0])
            pred_upper.append(conf[0][1])
            pred_lower.append(conf[0][0])
            model.update([fc[0]])
    return pd.Series(y_pred, index=index), pred_upper, pred_lower

# Preprocess Seoul data
seoul_apple = dataloader_KAMIS.seoul_apple
seoul_apple_price = pd.Series(preprocess_prices(seoul_apple['price']), index=pd.to_datetime(seoul_apple['date']))
seoul_baechu = dataloader_KAMIS.seoul_baechu
seoul_baechu_price = pd.Series(preprocess_prices(seoul_baechu['price']), index=pd.to_datetime(seoul_baechu['date']))
seoul_fish = dataloader_KAMIS.seoul_fish
seoul_fish_price = pd.Series(preprocess_prices(seoul_fish['price']), index=pd.to_datetime(seoul_fish['date']))
seoul_pig = dataloader_KAMIS.seoul_pig
seoul_pig_price = pd.Series(preprocess_prices(seoul_pig['price']), index=pd.to_datetime(seoul_pig['date']))
seoul_rice = dataloader_KAMIS.seoul_rice
seoul_rice_price = pd.Series(preprocess_prices(seoul_rice['price']), index=pd.to_datetime(seoul_rice['date']))

# Preprocess Busan data
busan_apple = dataloader_KAMIS.busan_apple
busan_apple_price = pd.Series(preprocess_prices(busan_apple['price']), index=pd.to_datetime(busan_apple['date']))
busan_baechu = dataloader_KAMIS.busan_baechu
busan_baechu_price = pd.Series(preprocess_prices(busan_baechu['price']), index=pd.to_datetime(busan_baechu['date']))
busan_fish = dataloader_KAMIS.busan_fish
busan_fish_price = pd.Series(preprocess_prices(busan_fish['price']), index=pd.to_datetime(busan_fish['date']))
busan_pig = dataloader_KAMIS.busan_pig
busan_pig_price = pd.Series(preprocess_prices(busan_pig['price']), index=pd.to_datetime(busan_pig['date']))
busan_rice = dataloader_KAMIS.busan_rice
busan_rice_price = pd.Series(preprocess_prices(busan_rice['price']), index=pd.to_datetime(busan_rice['date']))

# Preprocess Daegu data
daegu_apple = dataloader_KAMIS.daegu_apple
daegu_apple_price = pd.Series(preprocess_prices(daegu_apple['price']), index=pd.to_datetime(daegu_apple['date']))
daegu_baechu = dataloader_KAMIS.daegu_baechu
daegu_baechu_price = pd.Series(preprocess_prices(daegu_baechu['price']), index=pd.to_datetime(daegu_baechu['date']))
daegu_fish = dataloader_KAMIS.daegu_fish
daegu_fish_price = pd.Series(preprocess_prices(daegu_fish['price']), index=pd.to_datetime(daegu_fish['date']))
daegu_pig = dataloader_KAMIS.daegu_pig
daegu_pig_price = pd.Series(preprocess_prices(daegu_pig['price']), index=pd.to_datetime(daegu_pig['date']))
daegu_rice = dataloader_KAMIS.daegu_rice
daegu_rice_price = pd.Series(preprocess_prices(daegu_rice['price']), index=pd.to_datetime(daegu_rice['date']))

# List of data sets and titles
data_list = [
    (seoul_apple_price, "Seoul Apple Price"),
    (seoul_baechu_price, "Seoul Baechu Price"),
    (seoul_fish_price, "Seoul Fish Price"),
    (seoul_pig_price, "Seoul Pig Price"),
    (seoul_rice_price, "Seoul Rice Price"),
    (busan_apple_price, "Busan Apple Price"),
    (busan_baechu_price, "Busan Baechu Price"),
    (busan_fish_price, "Busan Fish Price"),
    (busan_pig_price, "Busan Pig Price"),
    (busan_rice_price, "Busan Rice Price"),
    (daegu_apple_price, "Daegu Apple Price"),
    (daegu_baechu_price, "Daegu Baechu Price"),
    (daegu_fish_price, "Daegu Fish Price"),
    (daegu_pig_price, "Daegu Pig Price"),
    (daegu_rice_price, "Daegu Rice Price")
]

# Iterate over each data set
for price_series, title in data_list:
    fc, upper, lower, future_forecast, future_upper, future_lower, price_index, future_index = fit_and_forecast(price_series)
    plot_forecast(price_series, fc, upper, lower, future_forecast, future_upper, future_lower, future_index, title)
